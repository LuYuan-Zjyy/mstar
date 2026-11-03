"""Request state kept on the device across decode steps.

Two steps of one batch differ by "every length +1" and "a row crossed a page
boundary", yet a host plan rebuilds each row's write location, position and
recurrent slot every step, and the plan thread shares a core with the other two
step threads. Here each request holds a slot, a row in fixed-address tables on
the device. The owner of a state keeps its own table (the KV cache its block
table and lengths, the recurrent pool its slot, positions their offset), and a
captured step reads them by a per-step row of slots, updates them itself, and
needs from the host only what changed.

Every table write lands on the node's stream in step order: the graph's own
scatters, and the row copies `ResidentPlanner.flush` issues right before a
replay. The plan stream only ever writes a step's own staging buffers. So a row
is never rewritten under a graph that reads it, an abandoned pre-plan leaves the
tables alone, and nothing on the host path needs a kernel: a slot's fields sit
side by side in one row of one store, so rewriting it is one H2D.

The host stays the source of truth. A row is written whole the first time a
step reads it and whenever something other than a resident commit changed it
(a host-planned step, an offload, a reload, a fork); see `mark_stale`.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from itertools import chain

import numpy as np
import torch
import triton
import triton.language as tl

from mstar.engine.resources.base import Resource
from mstar.engine.resources.step import BucketKey, SubmoduleStep
from mstar.utils.h2d import PinnedStager
from mstar.utils.profiler import PhaseLaps

logger = logging.getLogger(__name__)

# Row 0 of every table, never a request's. Padding rows read and write it: its
# cells stay zero, which every owner's table takes to mean the sink.
TRASH_SLOT = 0


# What a captured step records: a small kernel apiece. The torch ops for the same
# work take several times the GPU time, and the replay's launch on the GPU thread.

@triton.jit
def _scatter_cells_kernel(cells_ptr, values_ptr, store_ptr):
    i = tl.program_id(0)
    tl.store(store_ptr + tl.load(cells_ptr + i), tl.load(values_ptr + i))


@triton.jit
def _gather_column_kernel(
    slots_ptr, store_ptr, out_ptr, plus_ptr, STRIDE: tl.constexpr, COL: tl.constexpr, PLUS: tl.constexpr,
):
    i = tl.program_id(0)
    value = tl.load(store_ptr + tl.load(slots_ptr + i) * STRIDE + COL)
    if PLUS:
        value += tl.load(plus_ptr + i)
    tl.store(out_ptr + i, value.to(out_ptr.dtype.element_ty))


@triton.jit
def _advance_column_kernel(slots_ptr, store_ptr, STRIDE: tl.constexpr, COL: tl.constexpr, TRASH: tl.constexpr):
    i = tl.program_id(0)
    slot = tl.load(slots_ptr + i)
    cell = slot * STRIDE + COL
    tl.store(store_ptr + cell, tl.load(store_ptr + cell) + tl.where(slot != TRASH, 1, 0))


def scatter_cells(store: torch.Tensor, cells: torch.Tensor) -> None:
    """``store[cells[0]] = cells[1]``."""
    if store.is_cuda:
        _scatter_cells_kernel[(cells.shape[1],)](cells[0], cells[1], store, num_warps=1)
    else:
        store.index_copy_(0, cells[0], cells[1])


def gather_column(
    store: torch.Tensor, slots: torch.Tensor, stride: int, col: int, out: torch.Tensor,
    plus: torch.Tensor | None = None,
) -> None:
    """``out[i] = store[slots[i], col]``, plus ``plus[i]``."""
    n = slots.numel()
    if store.is_cuda:
        _gather_column_kernel[(n,)](
            slots, store, out, plus if plus is not None else slots,
            STRIDE=stride, COL=col, PLUS=plus is not None, num_warps=1,
        )
        return
    value = store.index_select(0, slots * stride + col)
    out[:n].copy_(value if plus is None else value + plus)


def advance_column(store: torch.Tensor, slots: torch.Tensor, stride: int, col: int) -> None:
    """``store[slots[i], col] += 1`` for every row but the trash row's."""
    if store.is_cuda:
        _advance_column_kernel[(slots.numel(),)](slots, store, STRIDE=stride, COL=col, TRASH=TRASH_SLOT, num_warps=1)
    else:
        cells = slots * stride + col
        store.index_copy_(0, cells, store.index_select(0, cells) + (slots != TRASH_SLOT))


class ResidentTable:
    """``width`` columns of the planner's store, one cell per slot per column."""

    __slots__ = ("col", "width", "stride", "view")

    def __init__(self, width: int):
        self.width = width
        # set when the planner seals the store
        self.col = 0
        self.stride = 0
        self.view: torch.Tensor | None = None  # `[slots, width]`

    def cell(self, slot: int, col: int = 0) -> int:
        """A column's cell in the flat store."""
        return slot * self.stride + self.col + col


class ResidentWrites:
    """What one step changes. ``runs`` are (first cell, values), rewritten whole
    by `flush`; ``cells`` and ``values`` are single cells, which ride the graph's
    own scatter. ``labels`` is each owner's stream this step and ``reads`` the
    owners that only read theirs, both by resource key."""

    __slots__ = ("labels", "reads", "runs", "cells", "values", "kv_len")

    def __init__(self, labels: Mapping[str, str], reads: frozenset[str]):
        self.labels = labels
        self.reads = reads
        self.runs: list[tuple[int, list[int]]] = []
        self.cells: list[int] = []
        self.values: list[int] = []
        # each KV pool's stored length of the row being rewritten, for the owners after it
        self.kv_len: dict[str, int] = {}

    def put(self, cell: int, value: int) -> None:
        self.cells.append(cell)
        self.values.append(value)

    def run(self, cell: int, values: list[int]) -> None:
        self.runs.append((cell, values))


@dataclass(slots=True)
class ResidentBuffers:
    """What one (bucket, slot) capture reads per step. ``cells`` is `[2, 2 * bs]`,
    flat cells then values, so a step's single-cell writes are one copy: a row
    has at most a length and a new column to write per KV pool. Unused entries
    write zero to cell zero, a trash cell, and ``dirty`` says whether any entry
    is in use."""
    slots: torch.Tensor
    cells: torch.Tensor
    # each KV pool's stored lengths of the rows this step, by resource key: the
    # graph writes them, for what reads them after
    kv_len: dict[str, torch.Tensor]
    # the host row ``slots`` holds: a steady batch keeps one array, so no H2D and no compare
    staged: np.ndarray | None = None
    dirty: bool = False


@dataclass(slots=True)
class ResidentPlan:
    """One step's row of slots and what it changes."""
    key: tuple
    rids: tuple[int, ...]
    slots: np.ndarray
    seq: int
    commits: int
    writes: ResidentWrites
    # rows written whole by this plan, and rows whose length alone it refreshes
    taken: set[int]
    moved: set[int]
    # (rows, pages after the step) of each KV pool's crossings, for `resident_advance`
    crossed: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)


@dataclass(slots=True)
class CapturedResidentStep:
    """What one capture's kernels share, at warmup and capture only."""
    bucket: BucketKey
    slot: int
    slots: torch.Tensor
    buffers: ResidentBuffers
    # each owner's stream label, and the owners that only read theirs, by resource key
    labels: Mapping[str, str]
    reads: frozenset[str]
    store: torch.Tensor | None = None
    # each KV pool's block table, by resource key, for attention that reads a row's pages itself
    block_tables: dict[str, ResidentTable] = field(default_factory=dict)
    # KV pools whose attention runs off the tables alone: nothing plans them on this bucket
    host_plan_free: set[str] = field(default_factory=set)
    # KV pools whose attention builds its page arrays in the graph: the host plans lengths only
    lengths_only: set[str] = field(default_factory=set)


class ResidentPlanner:
    """One node's request slots, tables and per-step rows."""

    def __init__(
        self, node: str, walks: Collection[str], resources: Mapping[str, Resource],
        device: torch.device, max_slots: int,
    ):
        self.node = node
        self.walks = frozenset(walks)
        self.device = device
        # request rows plus the trash row
        self.max_slots = max_slots + 1
        self._owners = dict(resources)
        # the KV pools' resource keys, as they bind; each has a block table and lengths
        self.kv_pools: list[str] = []
        self._tables: list[ResidentTable] = []
        self._store: torch.Tensor | None = None
        # columns of the one-cell tables, which a rewritten row always covers
        self.narrow_cols = 0
        self._stager = PinnedStager(torch.long, numel=1 << 16)
        self._buffers: dict[tuple[BucketKey, int], ResidentBuffers] = {}
        # (bucket, KV pool) whose steps need no host plan of the pool at all
        self.host_plan_free: set[tuple[BucketKey, str]] = set()
        # (bucket, KV pool) whose attention reads its pages off the tables: the
        # pool's plan hands it host lengths and no page indices
        self.lengths_only: set[tuple[BucketKey, str]] = set()
        # bucket -> (labels, reads) of its steps, from its capture
        self._streams_of: dict[BucketKey, tuple[dict[str, str], frozenset[str]]] = {}

        self._lock = threading.Lock()
        # rid -> slot. Handle-keyed: `release` purges it.
        self._slot_of: dict[int, int] = {}
        self._free: list[int] = list(range(max_slots, TRASH_SLOT, -1))
        # (slot, seq of the last step that read it), freed under a step not yet committed
        self._cooling: list[tuple[int, int]] = []
        self._last_read = np.zeros(self.max_slots, dtype=np.int64)
        # Handle-keyed, like `_slot_of`. Rows to write whole, and rows a
        # host-planned step only advanced as a resident one would have.
        self._stale: set[int] = set()
        self._moved: set[int] = set()
        self._plan_seq = 0
        self._committed_seq = 0
        self._commits = 0
        self._staged: ResidentPlan | None = None
        # the last batch's rows, so a steady batch skips the per-row lookup
        self._last_rids: tuple[int, ...] | None = None
        self._last_slots: np.ndarray | None = None

        for key, resource in self._owners.items():
            resource.bind_resident(self, key)
        self._seal()

    # Tables

    def table(self, width: int = 1) -> ResidentTable:
        """A table of ``width`` columns. Only from `bind_resident`."""
        if self._store is not None:
            raise RuntimeError("resident tables are fixed once the planner is built")
        table = ResidentTable(width)
        self._tables.append(table)
        return table

    def _seal(self) -> None:
        # the narrow tables first, so a rewritten row's narrow cells are one run
        wide = [t for t in self._tables if t.width > 1]
        narrow = [t for t in self._tables if t.width == 1]
        self.narrow_cols = len(narrow)
        stride = self._stride = sum(t.width for t in self._tables)
        # int64: what the KV write addressing and the positions are, and index ops take
        self._store = torch.zeros(self.max_slots * stride, dtype=torch.long, device=self.device)
        rows = self._store.view(self.max_slots, stride)
        col = 0
        for t in (*narrow, *wide):
            t.col, t.stride = col, stride
            t.view = rows[:, col: col + t.width]
            col += t.width
        logger.info(
            "resident planning on %s: %d slots of %d cells, %.1f MiB, walks %s",
            self.node, self.max_slots - 1, stride, self._store.nbytes / 2**20, sorted(self.walks),
        )

    @property
    def store(self) -> torch.Tensor:
        return self._store

    def covers(self, bucket: BucketKey) -> bool:
        return bucket.graph_walk in self.walks

    # Slots

    def slot_of(self, rid: int) -> int | None:
        return self._slot_of.get(rid)

    def _take_slot(self) -> int:
        """Caller holds the lock. A freed slot comes back once the last step
        that read it has committed: until then that step's commit still
        advances the slot's host lengths."""
        if self._cooling:
            done = self._committed_seq
            ready = [slot for slot, seq in self._cooling if seq <= done]
            if ready:
                self._cooling = [(slot, seq) for slot, seq in self._cooling if seq > done]
                self._free.extend(ready)
        if not self._free:
            raise RuntimeError(
                f"resident planning on {self.node}: all {self.max_slots - 1} slots are held; "
                "raise `knobs: resident_slots`"
            )
        return self._free.pop()

    def release(self, rid: int) -> None:
        """The request is gone. Its slot cools until the last step that read it
        has committed; on the device the row is only read again behind that
        step, by stream order."""
        with self._lock:
            self._stale.discard(rid)
            self._moved.discard(rid)
            slot = self._slot_of.pop(rid, None)
            if slot is None:
                return
            # a handle comes back as another request, under a batch tuple equal to the old one
            self._last_rids = None
            if self._staged is not None and rid in self._staged.rids:
                self._staged.commits = -1
            seq = int(self._last_read[slot])
            if seq <= self._committed_seq:
                self._free.append(slot)
            else:
                self._cooling.append((slot, seq))

    def mark_stale(self, rid: int) -> None:
        """Something other than a resident commit changed ``rid``'s row."""
        if rid in self._slot_of:
            with self._lock:
                self._stale.add(rid)

    # The step

    def buffers(self, bucket: BucketKey, slot: int) -> ResidentBuffers:
        """The per-step rows of one capture, every slot at the trash row until staged."""
        key = (bucket, slot)
        found = self._buffers.get(key)
        if found is None:
            n = 2 * bucket.bs * max(1, len(self.kv_pools))
            found = self._buffers[key] = ResidentBuffers(
                slots=torch.full((bucket.bs,), TRASH_SLOT, dtype=torch.long, device=self.device),
                cells=torch.zeros((2, n), dtype=torch.long, device=self.device),
                kv_len={
                    pool: torch.zeros(bucket.bs, dtype=torch.long, device=self.device)
                    for pool in self.kv_pools
                },
            )
        return found

    def plan(self, step: SubmoduleStep, ctx) -> None:
        """Stage this step's rows; on the plan thread for a pre-plan. Its admit
        has run, so every page the step writes is on its stream."""
        lease = ctx.slot_lease
        rids = tuple(ctx.request_ids)
        key = (rids, lease.bucket, lease.slot)
        staged = self._staged
        if staged is not None:
            self._staged = None
            if not ctx.is_preplan and staged.key == key and staged.commits == self._commits:
                ctx.resident_plan = staged
                return
            # the step it was for never ran, or something ran first
            self._restore(staged)
        plan = self._build(rids, key, self._streams(lease.bucket, step), ctx)
        ctx.resident_plan = plan
        if ctx.is_preplan:
            self._staged = plan

    def drop_stage(self) -> None:
        """The pages a pre-plan's stage names were given back (a refused admit,
        an abandoned pre-plan): a retry over the same rows reserves its own, so
        it plans afresh instead of promoting cells that name the old ones."""
        staged, self._staged = self._staged, None
        if staged is not None:
            self._restore(staged)

    def _restore(self, plan: ResidentPlan) -> None:
        """Hand back the rows of a plan that won't run: it wrote nothing."""
        with self._lock:
            self._stale |= {rid for rid in plan.taken if rid in self._slot_of}
            self._moved |= {rid for rid in plan.moved if rid in self._slot_of}

    def _build(
        self, rids: tuple[int, ...], key: tuple, streams: tuple[dict[str, str], frozenset[str]], ctx,
    ) -> ResidentPlan:
        lease = ctx.slot_lease
        laps = PhaseLaps(f"{ctx.plan_phase}.resident.")
        with self._lock:
            if rids == self._last_rids:
                slots = self._last_slots
            else:
                get = self._slot_of.get
                row = [get(rid) for rid in rids]
                if None in row:
                    for i, slot in enumerate(row):
                        if slot is None:
                            row[i] = self._slot_of[rids[i]] = self._take_slot()
                            self._stale.add(rids[i])
                slots = np.array(row, dtype=np.int64)
                self._last_rids, self._last_slots = rids, slots
            stale = self._stale
            taken = stale.intersection(rids) if stale else set()
            stale -= taken
            moved = self._moved
            refreshed = moved.intersection(rids) if moved else set()
            moved -= refreshed
            refreshed -= taken
            self._plan_seq += 1
            seq = self._plan_seq
            self._last_read[slots] = seq
            rows = [(rid, self._slot_of[rid]) for rid in taken]
        plan = ResidentPlan(key, rids, slots, seq, self._commits, ResidentWrites(*streams), taken, refreshed)
        laps.lap("rows")
        for rid, slot in rows:
            self.rewrite(rid, slot, plan.writes)
        laps.lap("rewrites")
        buffers = self.buffers(lease.bucket, lease.slot)
        if buffers.staged is not slots:
            self._stager.copy_(buffers.slots, slots, pad_value=TRASH_SLOT)
            buffers.staged = slots
        laps.lap("slot_row")
        for resource in self._owners.values():
            resource.resident_stage(plan)
        laps.lap("stage")
        self._stage_cells(plan, buffers)
        laps.lap("cells")
        return plan

    def rewrite(self, rid: int, slot: int, writes: ResidentWrites) -> None:
        """Queue ``rid``'s row whole: the narrow columns as one run, and each
        wide table's used cells as a run its owner queues."""
        row = [0] * self.narrow_cols
        writes.run(slot * self._stride, row)
        for resource in self._owners.values():
            resource.resident_row(rid, slot, row, writes)

    def _stage_cells(self, plan: ResidentPlan, buffers: ResidentBuffers) -> None:
        """One copy, on the plan stream: the graph scatters these first thing."""
        cells = plan.writes.cells
        if not cells and not buffers.dirty:
            return
        n = buffers.cells.shape[1]
        assert len(cells) <= n, f"resident planning on {self.node}: {len(cells)} cell writes, room for {n}"
        staged = np.zeros(2 * n, dtype=np.int64)
        staged[: len(cells)] = cells
        staged[n: n + len(cells)] = plan.writes.values
        buffers.dirty = bool(cells)
        self._stager.copy_(buffers.cells, staged)

    def flush(self, plan: ResidentPlan) -> None:
        """The rows `plan` rewrites, on the step's stream: behind every earlier
        step, ahead of this one. On the GPU thread, right before the replay."""
        runs = plan.writes.runs
        if not runs:
            return
        # a copy per contiguous span: a row's narrow cells and its first block table are one
        spans: list[list[int]] = []
        for cell, values in runs:
            if spans and spans[-1][1] == cell:
                spans[-1][1] += len(values)
            else:
                spans.append([cell, cell + len(values)])
        dsts = [self._store[start:end] for start, end in spans]
        self._stager.copy_runs_(dsts, np.fromiter(chain.from_iterable(v for _, v in runs), dtype=np.int64))

    def commit(self, step: SubmoduleStep, ctx) -> None:
        """After the resources' commit, before the step's commit event: the
        next pre-plan reads what this leaves."""
        plan = ctx.resident_plan if ctx.resident is not None else None
        self._commits += 1
        if plan is None:
            self._note_host_step(step, ctx)
            return
        for resource in self._owners.values():
            resource.resident_advance(plan)
        self._committed_seq = plan.seq

    def _note_host_step(self, step: SubmoduleStep, ctx) -> None:
        """A host-planned step moved its rows without their tables. A
        one-token row moved as a resident step would have, and needs only its
        length and any new column; anything else is written whole."""
        held = self._slot_of
        if not held:
            return
        segments = step.segments or [seg for s in step.steps.values() for seg in s.segments or ()]
        one = [seg.request_id for seg in segments if seg.span == 1 and seg.request_id in held]
        rest = [seg.request_id for seg in segments if seg.span != 1 and seg.request_id in held]
        with self._lock:
            self._moved.update(one)
            self._stale.update(rest)

    # The capture

    def _streams(self, bucket: BucketKey, step: SubmoduleStep) -> tuple[dict[str, str], frozenset[str]]:
        """Each owner's stream label on ``bucket``'s steps, and the owners that
        only read theirs (a zero span), by resource key; fixed by its capture."""
        found = self._streams_of.get(bucket)
        if found is not None:
            return found
        labels: dict[str, str] = {}
        reads = set()
        for key in self._owners:
            resource_step = step.steps.get(key)
            segments = (resource_step.segments if resource_step is not None else None) or step.segments
            if not segments:
                continue
            names = {seg.label for seg in segments}
            spans = {seg.span for seg in segments}
            if len(names) != 1 or spans not in ({0}, {1}):
                raise ValueError(
                    f"resident walk {bucket.graph_walk!r} on {self.node}: resource {key!r} declares "
                    f"{len(segments)} segments over labels {sorted(names)} and spans {sorted(spans)}; "
                    "a resident step is one stream a resource, one token a row or none"
                )
            labels[key] = names.pop()
            if spans == {0}:
                reads.add(key)
        found = self._streams_of[bucket] = (labels, frozenset(reads))
        return found

    def capture_step(self, bucket: BucketKey, slot: int, step: SubmoduleStep) -> CapturedResidentStep:
        """What a capture of ``bucket`` records around its forward. Refuses a
        walk whose declared step isn't one token per row, each row advancing
        alone, with each resource on one stream that it appends to or only
        reads: a captured step can't hold anything else."""
        if step is None:
            raise ValueError(f"resident walk {bucket.graph_walk!r} on {self.node} declares no step")
        segments = step.segments or next(iter(step.steps.values())).segments
        if any(seg.span != 1 for seg in segments) or len({seg.label for seg in segments}) != 1:
            raise ValueError(
                f"resident walk {bucket.graph_walk!r} on {self.node} declares "
                f"{len(segments)} default segments; a resident step is one token per row in one stream"
            )
        self._streams_of.pop(bucket, None)
        labels, reads = self._streams(bucket, step)
        for key, resource_step in step.steps.items():
            refusal = next(
                (
                    name for name in ("pre_forks", "post_forks", "retention", "sink", "advance", "pos_ids", "fork_rids")
                    if getattr(resource_step, name, None)
                ),
                None,
            )
            if refusal is not None:
                raise ValueError(
                    f"resident walk {bucket.graph_walk!r} on {self.node}: resource {key!r} declares "
                    f"{refusal}, which a resident step can't carry"
                )
        buffers = self.buffers(bucket, slot)
        return CapturedResidentStep(bucket, slot, buffers.slots, buffers, labels, reads)

    def gather(self, step: CapturedResidentStep) -> None:
        step.store = self._store
        # before the reads: a row that crossed writes into its new page
        scatter_cells(self._store, step.buffers.cells)
        for resource in self._owners.values():
            resource.resident_gather(step)
        for pool in step.host_plan_free:
            self.host_plan_free.add((step.bucket, pool))
        for pool in step.lengths_only - step.host_plan_free:
            self.lengths_only.add((step.bucket, pool))

    def commit_in_graph(self, step: CapturedResidentStep) -> None:
        for resource in self._owners.values():
            resource.resident_commit(step)


def build_planner(
    node: str, walks: Iterable[str], resources: Mapping[str, Resource],
    resource_nodes: Mapping[str, Collection[str]], device: torch.device, max_slots: int,
) -> tuple[ResidentPlanner | None, str]:
    """A planner for ``node``, or None and why not. Every resource of the node
    must be able to, since a capture is one graph: one host-planned resource
    would need the row a resident step no longer stages."""
    walks = frozenset(walks)
    if not walks:
        return None, "its submodule names no resident walk"
    for key, resource in resources.items():
        if len(resource_nodes.get(key, ())) > 1:
            return None, f"resource {key!r} is shared by nodes {sorted(resource_nodes[key])}"
        refusal = resource.resident_refusal()
        if refusal is not None:
            return None, f"resource {key!r}: {refusal}"
    return ResidentPlanner(node, walks, resources, device, max_slots), ""
