"""Right-angled enclosures found by CP-SAT on a lattice laid over the room.

The room is covered by a square lattice, turned to line up with one of its walls.
A candidate side runs from one lattice node to another along a lattice line, at a
length the inventory can build. CP-SAT then picks the sides:

- `add_circuit` makes the chosen sides one closed loop (a node can stay off it);
- a node is the start of at most one side or is crossed by at most one, so the
  loop never touches itself, and no two sides along one axis meet at a node, so
  every corner is a real 90-degree turn;
- every lattice cell is inside or outside the loop, tied to the sides around it
  (walking a side, the inside is on the left). Area is then the number of cells
  inside, which bounds the search much more tightly than the shoelace sum;
- the sides of each length are split among the ways the inventory can build that
  length (`Assigner.compositions`), so piece counts, straight connectors and gates
  are exact, and the second-gate bonus is a variable of its own.

Only sides that lie inside the room exist in the model, and the room is a simple
polygon, so whatever loop comes out encloses only room. The simulator still has
the last word on every construction.
"""

import math
from dataclasses import dataclass

import numpy as np
import shapely
from shapely import affinity

import src.constants as c
from players.player10.assign import Assigner
from src.enclosure import Construction
from src.pieces import Connector, ConnectorType, Piece, PieceType

try:
    from ortools.sat.python import cp_model
except ImportError:  # the stage is skipped without OR-Tools
    cp_model = None

AVAILABLE = cp_model is not None

DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))  # east, north, west, south
SCALE = 100  # objective coefficients are scaled to integers by this much


@dataclass
class Lattice:
    """The candidate sides of one lattice: nodes (i, j) sit at origin + (i, j) *
    spacing in a frame turned by `heading` degrees."""

    heading: float
    spacing: int
    origin: tuple[float, float]
    lo: tuple[int, int]  # grid index of node [0, 0]
    inside: np.ndarray  # [nx, ny] node lies in the room
    cells: np.ndarray  # [nx - 1, ny - 1] cell lies in the room
    sides: list[tuple[int, int, int, int, int]]  # (i, j, direction, steps, length)

    @property
    def nodes(self) -> int:
        return int(self.inside.sum())

    def to_world(self, i: int, j: int) -> tuple[float, float]:
        x = (self.lo[0] + i) * self.spacing
        y = (self.lo[1] + j) * self.spacing
        t = math.radians(self.heading)
        return (
            self.origin[0] + x * math.cos(t) - y * math.sin(t),
            self.origin[1] + x * math.sin(t) + y * math.cos(t),
        )


def count_nodes(room_polygon, heading: float, spacing: int, origin) -> int:
    """How many lattice nodes fall in the room: cheap, to size a lattice first."""
    local = local_room(room_polygon, heading, origin)
    gx, gy = _grid(local, spacing)
    region = local.buffer(c.TOL)
    return int(shapely.covers(region, shapely.points(gx * spacing, gy * spacing)).sum())


def local_room(room_polygon, heading: float, origin):
    """The room in the lattice frame: origin at (0, 0), lattice lines on the axes."""
    moved = affinity.translate(room_polygon, -origin[0], -origin[1])
    return affinity.rotate(moved, -heading, origin=(0, 0))


def _grid(local, spacing: int):
    x0, y0, x1, y1 = local.bounds
    i = np.arange(math.floor(x0 / spacing), math.ceil(x1 / spacing) + 1)
    j = np.arange(math.floor(y0 / spacing), math.ceil(y1 / spacing) + 1)
    return np.meshgrid(i, j, indexing="ij")


def build_lattice(
    room_polygon, heading: float, spacing: int, origin, lengths: list[int]
) -> Lattice:
    local = local_room(room_polygon, heading, origin)
    region = local.buffer(c.TOL)
    shapely.prepare(region)
    gi, gj = _grid(local, spacing)
    lo = (int(gi[0, 0]), int(gj[0, 0]))
    xs, ys = gi * spacing, gj * spacing
    inside = shapely.covers(region, shapely.points(xs, ys))
    cells = shapely.contains(
        region,
        shapely.box(xs[:-1, :-1], ys[:-1, :-1], xs[1:, 1:], ys[1:, 1:]),
    )

    nx, ny = inside.shape
    candidates = []
    usable = [(L, L // spacing) for L in lengths if L % spacing == 0]
    for i, j in zip(*np.nonzero(inside)):
        for d, (dx, dy) in enumerate(DIRS):
            for length, steps in usable:
                i2, j2 = i + dx * steps, j + dy * steps
                if 0 <= i2 < nx and 0 <= j2 < ny and inside[i2, j2]:
                    candidates.append((int(i), int(j), d, steps, length))
    sides = []
    if candidates:
        segments = shapely.linestrings(
            [
                [
                    (xs[i, j], ys[i, j]),
                    (xs[i, j] + DIRS[d][0] * L, ys[i, j] + DIRS[d][1] * L),
                ]
                for i, j, d, _, L in candidates
            ]
        )
        fits = shapely.contains(region, segments)
        sides = [s for s, ok in zip(candidates, fits) if ok]
    return Lattice(heading, spacing, origin, lo, inside, cells, sides)


@dataclass
class Found:
    construction: Construction
    model_score: float  # what the model expects the simulator to score
    proven: bool  # nothing better exists on this lattice
    # layered lattices only: the loop's corners in the lattice frame and the
    # layers it passes through, to refine it on a finer lattice
    corners: list[tuple[float, float]] | None = None
    layers: set[tuple[int, int]] | None = None


def solve(
    lattice: Lattice,
    assigner: Assigner,
    weights,
    right_connectors: int,
    beat: float,
    seconds: float,
) -> tuple[Found | None, str]:
    """The best loop on the lattice that scores above `beat`, if one exists and is
    found within `seconds` of CPU. Returns (found or None, solver status name)."""
    sides = lattice.sides
    if len(sides) < 4 or seconds <= 0:
        return None, "EMPTY"
    model = cp_model.CpModel()
    use = [model.new_bool_var(f"s{k}") for k in range(len(sides))]

    nx, ny = lattice.inside.shape
    node = -np.ones((nx, ny), dtype=int)
    node[lattice.inside] = np.arange(lattice.nodes)
    arcs = []
    starts_or_crosses = [[] for _ in range(lattice.nodes)]
    along = [([], []) for _ in range(lattice.nodes)]  # per axis
    h_edges: dict[tuple[int, int], list] = {}  # unit edge (i,j)-(i+1,j) -> terms
    v_edges: dict[tuple[int, int], list] = {}  # unit edge (i,j)-(i,j+1) -> terms
    for k, (i, j, d, steps, _) in enumerate(sides):
        dx, dy = DIRS[d]
        a, b = node[i, j], node[i + dx * steps, j + dy * steps]
        arcs.append((a, b, use[k]))
        starts_or_crosses[a].append(use[k])
        along[a][d % 2].append(use[k])
        along[b][d % 2].append(use[k])
        for t in range(1, steps):
            starts_or_crosses[node[i + dx * t, j + dy * t]].append(use[k])
        for t in range(steps):
            p, q = i + dx * t, j + dy * t
            if d == 0:
                h_edges.setdefault((p, q), []).append(use[k])
            elif d == 2:
                h_edges.setdefault((p - 1, q), []).append(-use[k])
            elif d == 1:
                v_edges.setdefault((p, q), []).append(use[k])
            else:
                v_edges.setdefault((p, q - 1), []).append(-use[k])
    arcs += [(n, n, model.new_bool_var(f"off{n}")) for n in range(lattice.nodes)]
    model.add_circuit(arcs)
    model.add(sum(use) >= 4)
    model.add(sum(use) <= right_connectors)  # one corner connector per side
    for n in range(lattice.nodes):
        if len(starts_or_crosses[n]) > 1:
            model.add_at_most_one(starts_or_crosses[n])
        for axis in along[n]:
            if len(axis) > 1:
                model.add(sum(axis) <= 1)

    # cells: inside on the left of every side walked
    cell = {
        (p, q): model.new_bool_var(f"c{p}_{q}")
        for p, q in zip(*np.nonzero(lattice.cells))
    }

    def inside(p, q):
        return cell.get((p, q), 0)

    for (p, q), terms in h_edges.items():
        model.add(inside(p, q) - inside(p, q - 1) == sum(terms))
    for (p, q), terms in v_edges.items():
        model.add(inside(p - 1, q) - inside(p, q) == sum(terms))
    # a cell whose edge no side can run along must match its neighbour across it
    for p in range(nx - 1):
        for q in range(ny):
            if (p, q) not in h_edges and ((p, q) in cell or (p, q - 1) in cell):
                model.add(inside(p, q) == inside(p, q - 1))
    for p in range(nx):
        for q in range(ny - 1):
            if (p, q) not in v_edges and ((p - 1, q) in cell or (p, q) in cell):
                model.add(inside(p - 1, q) == inside(p, q))

    inventory = add_inventory(model, [s[4] for s in sides], use, assigner, weights)
    if inventory is None:
        return None, "NO_GATE"
    counts, bonus = inventory

    area = lattice.spacing**2 * sum(cell.values())
    perimeter = sum(use[k] * s[4] for k, s in enumerate(sides))
    objective = (
        round(SCALE * weights.A) * area
        + round(SCALE * weights.C) * perimeter
        + round(SCALE * weights.G) * bonus
    )
    # only loops that beat what we already have are of any use
    model.add(objective >= math.floor(SCALE * (beat - c.BASELINE_SCORE)) + 1)
    model.maximize(objective)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_workers = 1  # CPU time of every thread counts
    status = solver.solve(model)
    name = solver.status_name(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, name

    chosen = {node[s[0], s[1]]: s for k, s in enumerate(sides) if solver.value(use[k])}
    first = next(iter(chosen))
    loop, at = [], first
    while True:
        i, j, d, steps, _ = chosen[at]
        loop.append(chosen[at])
        at = node[i + DIRS[d][0] * steps, j + DIRS[d][1] * steps]
        if at == first:
            break
    # right-angled: directions are quarter turns, doubled here into eighths
    pieces, connectors = build_pieces(
        [(2 * d, length) for _, _, d, _, length in loop], solver, counts
    )
    i, j, d, _, _ = loop[0]
    construction = Construction(
        lattice.to_world(i, j), (lattice.heading + 90.0 * d) % 360.0, pieces, connectors
    )
    score = c.BASELINE_SCORE + solver.objective_value / SCALE
    return Found(construction, score, status == cp_model.OPTIMAL), name


def add_inventory(model, lengths: list[int], use: list, assigner: Assigner, weights):
    """Constrain the chosen sides (`use[k]` has length `lengths[k]`) to what the
    inventory can build: the sides of each length are split among the ways to
    build that length, so piece counts, straight connectors and gates are exact.
    Returns (composition counts, second-gate bonus), or None if no gate fits."""
    by_length: dict[int, list] = {}
    for k, length in enumerate(lengths):
        by_length.setdefault(length, []).append(use[k])
    counts: dict[tuple, cp_model.IntVar] = {}
    piece_use = {key: [] for key in assigner.stock}
    straights, gated = [], []
    for length, chosen in by_length.items():
        plain, with_gate = assigner.compositions[length]
        split = []
        for comp in plain + with_gate:
            n = model.new_int_var(0, len(chosen), "")
            counts[(length, comp)] = n
            split.append(n)
            for key in set(comp):
                piece_use[key].append(comp.count(key) * n)
            if len(comp) > 1:
                straights.append((len(comp) - 1) * n)
            if any(k[0] for k in comp):
                gated.append(n)
        model.add(sum(split) == sum(chosen))
    for key, terms in piece_use.items():
        if terms:
            model.add(sum(terms) <= assigner.stock[key])
    model.add(sum(straights) <= assigner.straights)
    if not gated:
        return None
    model.add(sum(gated) >= 1)
    bonus = 0
    if weights.G > 0 and assigner.gates >= 2:
        bonus = model.new_bool_var("bonus")
        model.add(sum(gated) >= 2).only_enforce_if(bonus)
    return counts, bonus


def build_pieces(loop: list[tuple[int, int]], solver, counts):
    """Pieces and connectors for a loop given as (direction in eighths of a turn,
    length) per side, walked counterclockwise, from the compositions the solver
    chose. The connector before a side's first piece makes its corner: 45 degrees
    takes a diagonal connector, 90 a right one; a right turn is a reflex corner."""
    pool: dict[int, list] = {}
    for (length, comp), n in counts.items():
        pool.setdefault(length, []).extend([comp] * solver.value(n))
    pieces, connectors = [], []
    for index, (d, length) in enumerate(loop):
        turn = (d - loop[index - 1][0] + 4) % 8 - 4  # in eighths, + is left
        kind = ConnectorType.DIAGONAL if abs(turn) == 1 else ConnectorType.RIGHT
        for q, (is_gate, piece_length) in enumerate(pool[length].pop()):
            piece_type = PieceType.GATE if is_gate else PieceType.WALL
            pieces.append(Piece(piece_type, piece_length))
            connectors.append(
                Connector(kind, reflex=turn < 0)
                if q == 0
                else Connector(ConnectorType.STRAIGHT)
            )
    return pieces, connectors


def loop_possible(assigner: Assigner, right_connectors: int, seconds: float) -> bool:
    """Whether the inventory alone can make any right-angled loop: sides that
    cancel east against west and north against south, with a gate. No geometry,
    so it answers in milliseconds and spares the lattice searches a hopeless case."""
    if right_connectors < 4 or assigner.gates == 0:
        return False
    model = cp_model.CpModel()
    piece_use = {key: [] for key in assigner.stock}
    straights, gated, total = [], [], {d: [] for d in range(4)}
    count = {d: [] for d in range(4)}
    for length in assigner.lengths:
        plain, with_gate = assigner.compositions[length]
        for comp in plain + with_gate:
            for d in range(4):
                n = model.new_int_var(0, right_connectors, "")
                for key in set(comp):
                    piece_use[key].append(comp.count(key) * n)
                if len(comp) > 1:
                    straights.append((len(comp) - 1) * n)
                if any(k[0] for k in comp):
                    gated.append(n)
                total[d].append(length * n)
                count[d].append(n)
    for key, terms in piece_use.items():
        if terms:
            model.add(sum(terms) <= assigner.stock[key])
    model.add(sum(straights) <= assigner.straights)
    model.add(sum(gated) >= 1)
    model.add(sum(total[0]) == sum(total[2]))
    model.add(sum(total[1]) == sum(total[3]))
    for d in range(4):
        model.add(sum(count[d]) >= 1)
    sides_h = sum(count[0]) + sum(count[2])
    sides_v = sum(count[1]) + sum(count[3])
    model.add(sides_h == sides_v)  # a right-angled loop alternates axes
    model.add(sides_h + sides_v <= right_connectors)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_workers = 1
    status = solver.solve(model)
    # unknown counts as possible: only a proof rules the stage out
    return status != cp_model.INFEASIBLE
