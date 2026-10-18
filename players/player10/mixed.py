"""Loops with 45- and 90-degree corners, found by CP-SAT on a layered lattice.

A diagonal side of whole length L moves L * sqrt(2)/2 along both axes, which is
never a whole number of lattice steps, so on a plain lattice its end lands
between nodes. The fix is to keep that drift exactly: a node is (i, j, a, b) at

    x = i * spacing + a * sqrt(2)/2,    y = j * spacing + b * sqrt(2)/2,

so (a, b) = (0, 0) is the plain lattice and every other (a, b) is a copy of it,
shifted by a whole number of half-diagonals: a "layer". Straight sides stay in
their layer; a diagonal side of length L moves to the layer (a +- L, b +- L). A
loop closes only if it comes back to its own node, so every diagonal drift is
cancelled by an opposite one -- the closing rule needs no extra constraint. Only
the layers within `moves` diagonal sides of (0, 0) exist, which keeps the
lattice small; with the few diagonal connectors scenarios hand out, a loop never
strays further than that from where its diagonals started.

What `circuit.py` gets for free does not hold here: sides in different layers
can cross between nodes. Rather than forbid every crossing pair up front, each
solution is checked, the pairs that actually cross are forbidden, and the model
is solved again (lazy constraints); the best loops rarely cross themselves, so
this takes a round or two. Area is the shoelace sum over the chosen sides,
required positive so the loop runs counterclockwise.
"""

import math
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import LineString

import src.constants as c
from players.player10.assign import Assigner
from players.player10.circuit import (
    SCALE,
    Found,
    add_inventory,
    build_pieces,
    cp_model,
    local_room,
)
from src.enclosure import Construction

HALF_ROOT2 = math.sqrt(2) / 2
# per direction (eighths of a turn from the lattice's +x axis): the lattice step
# a straight side takes per unit, and the layer step a diagonal takes per unit
GRID_STEP = [(1, 0), None, (0, 1), None, (-1, 0), None, (0, -1), None]
LAYER_STEP = [None, (1, 1), None, (-1, 1), None, (-1, -1), None, (1, -1)]
# turns (out - in, in eighths) no connector can make: straight on, 135 degrees
# either way, and turning back
FORBIDDEN_TURNS = {0, 3, 4, 5}
MAX_CUT_ROUNDS = 12


def layers(diagonals: list[int], moves: int) -> set[tuple[int, int]]:
    """Every (a, b) within `moves` diagonal sides of (0, 0)."""
    seen, frontier = {(0, 0)}, {(0, 0)}
    for _ in range(moves):
        reached = {
            (a + da * d, b + db * d)
            for a, b in frontier
            for d in diagonals
            for da, db in (LAYER_STEP[1], LAYER_STEP[3], LAYER_STEP[5], LAYER_STEP[7])
        }
        frontier = reached - seen
        seen |= reached
    return seen


@dataclass
class MixedLattice:
    heading: float
    spacing: int
    origin: tuple[float, float]
    points: np.ndarray  # [n, 2] node positions in the lattice frame
    # (from node, to node, direction in eighths, length, nodes passed through)
    sides: list[tuple[int, int, int, int, list[int]]]
    layer: list[tuple[int, int]] | None = None  # each node's (a, b), if layered
    # right-angled lattices only: the distinct node x and y coordinates, and
    # which of the rectangles between them lie in the room (see `_cell_area`)
    xs: np.ndarray | None = None
    ys: np.ndarray | None = None
    cells: np.ndarray | None = None

    def to_world(self, x: float, y: float) -> tuple[float, float]:
        t = math.radians(self.heading)
        return (
            self.origin[0] + x * math.cos(t) - y * math.sin(t),
            self.origin[1] + x * math.sin(t) + y * math.cos(t),
        )


def build_mixed_lattice(
    room_polygon,
    heading: float,
    spacing: int | tuple[int, int],
    origin,
    straight_lengths: list[int],
    diagonals: list[int],
    moves: int,
    max_nodes: int,
    only_layers: set[tuple[int, int]] | None = None,
) -> MixedLattice | None:
    """The layered lattice and every side that lies in the room, or None when it
    has more than `max_nodes` nodes. `spacing` may differ per axis, (x, y): a
    narrow room wants a fine step across it but can stay coarse along it.
    `only_layers` keeps just those layers: refining a loop found on a coarser
    lattice, only the layers it passes through are needed."""
    sx, sy = spacing if isinstance(spacing, tuple) else (spacing, spacing)
    local = local_room(room_polygon, heading, origin)
    region = local.buffer(c.TOL)
    shapely.prepare(region)
    x0, y0, x1, y1 = local.bounds
    keys, points = [], []
    for a, b in sorted(only_layers or layers(diagonals, moves)):
        dx, dy = a * HALF_ROOT2, b * HALF_ROOT2
        gi, gj = np.meshgrid(
            np.arange(math.floor((x0 - dx) / sx), math.ceil((x1 - dx) / sx) + 1),
            np.arange(math.floor((y0 - dy) / sy), math.ceil((y1 - dy) / sy) + 1),
            indexing="ij",
        )
        xs, ys = gi * sx + dx, gj * sy + dy
        inside = shapely.covers(region, shapely.points(xs, ys))
        for i, j, x, y in zip(gi[inside], gj[inside], xs[inside], ys[inside]):
            keys.append((int(i), int(j), a, b))
            points.append((float(x), float(y)))
        if len(keys) > max_nodes:
            return None
    index = {key: n for n, key in enumerate(keys)}

    candidates = []
    # whole lattice steps a straight side of each length takes, per axis
    steps_x = [(L, L // sx) for L in straight_lengths if L % sx == 0]
    steps_y = [(L, L // sy) for L in straight_lengths if L % sy == 0]
    for n, (i, j, a, b) in enumerate(keys):
        for d in range(8):
            if GRID_STEP[d]:
                di, dj = GRID_STEP[d]
                for length, steps in steps_x if di else steps_y:
                    end = index.get((i + di * steps, j + dj * steps, a, b))
                    if end is not None:
                        through = [
                            index[k]
                            for t in range(1, steps)
                            if (k := (i + di * t, j + dj * t, a, b)) in index
                        ]
                        candidates.append((n, end, d, length, through))
            else:
                da, db = LAYER_STEP[d]
                for length in diagonals:
                    end = index.get((i, j, a + da * length, b + db * length))
                    if end is not None:
                        through = [
                            index[k]
                            for t in range(1, length)
                            if (k := (i, j, a + da * t, b + db * t)) in index
                        ]
                        candidates.append((n, end, d, length, through))
    points = np.array(points)
    sides = []
    if candidates:
        segments = shapely.linestrings(
            [[points[s[0]], points[s[1]]] for s in candidates]
        )
        fits = shapely.contains(region, segments)
        sides = [s for s, ok in zip(candidates, fits) if ok]
    return MixedLattice(
        heading, sx, tuple(origin), points, sides, [k[2:] for k in keys]
    )


def solve_mixed(
    lattice: MixedLattice,
    assigner: Assigner,
    weights,
    right_connectors: int,
    diagonal_connectors: int,
    beat: float,
    seconds: float,
    hint: list[tuple[float, float]] | None = None,
) -> tuple[Found | None, str]:
    """The best simple loop on the lattice scoring above `beat`, if one is found
    within `seconds` of CPU in all. Returns (found or None, status name).

    `hint` is a loop to start from, as corners in the lattice frame: the best
    loop of a coarser lattice when refining it. The solver then only has to
    improve it, which on a big lattice it does far sooner than it finds a loop
    of its own."""
    sides, points = lattice.sides, lattice.points
    if len(sides) < 3 or seconds <= 0:
        return None, "EMPTY"
    nodes = len(points)
    model = cp_model.CpModel()
    use = [model.new_bool_var(f"s{k}") for k in range(len(sides))]

    arcs = [(s[0], s[1], use[k]) for k, s in enumerate(sides)]
    arcs += [(n, n, model.new_bool_var(f"off{n}")) for n in range(nodes)]
    model.add_circuit(arcs)
    model.add(sum(use) >= c.MIN_ENCLOSURE_PIECES)

    touching = [[] for _ in range(nodes)]
    arriving = [[[] for _ in range(8)] for _ in range(nodes)]
    leaving = [[[] for _ in range(8)] for _ in range(nodes)]
    for k, (start, end, d, _, through) in enumerate(sides):
        touching[start].append(use[k])
        for n in through:
            touching[n].append(use[k])
        leaving[start][d].append(use[k])
        arriving[end][d].append(use[k])
    diagonal_turns = []
    for n in range(nodes):
        if len(touching[n]) > 1:
            model.add_at_most_one(touching[n])
        for x in range(8):
            if not arriving[n][x]:
                continue
            for y in range(8):
                if leaving[n][y] and (y - x) % 8 in FORBIDDEN_TURNS:
                    model.add(sum(arriving[n][x]) + sum(leaving[n][y]) <= 1)
        # a corner takes a diagonal connector exactly when one of its two sides
        # is diagonal and the other is not
        into = sum(v for d in (1, 3, 5, 7) for v in arriving[n][d])
        out = sum(v for d in (1, 3, 5, 7) for v in leaving[n][d])
        if isinstance(into, int) and isinstance(out, int):
            continue
        t = model.new_bool_var(f"t{n}")
        model.add(t >= into - out)
        model.add(t >= out - into)
        model.add(t <= into + out)  # exactly one of the two, no more
        model.add(t <= 2 - into - out)
        diagonal_turns.append(t)
    model.add(sum(diagonal_turns) <= diagonal_connectors)
    model.add(sum(use) - sum(diagonal_turns) <= right_connectors)

    inventory = add_inventory(model, [s[3] for s in sides], use, assigner, weights)
    if inventory is None:
        return None, "NO_GATE"
    counts, bonus = inventory

    if lattice.cells is not None:
        area = _cell_area(model, lattice, use)
    else:
        twice_area = [
            points[s[0]][0] * points[s[1]][1] - points[s[1]][0] * points[s[0]][1]
            for s in sides
        ]
        area = sum(use[k] * round(SCALE * twice_area[k] / 2) for k in range(len(sides)))
    model.add(area >= SCALE)  # counterclockwise, and not empty
    perimeter = sum(use[k] * s[3] for k, s in enumerate(sides))
    objective = (
        round(SCALE * weights.A) * area
        + round(SCALE * SCALE * weights.C) * perimeter
        + round(SCALE * SCALE * weights.G) * bonus
    )
    # only loops that beat what we already have are of any use; a hint scores
    # about that, so then the bar is lowered a little to let the solver start
    hinted = hint is not None and _add_hint(model, use, sides, points, hint)
    floor = math.floor(SCALE * SCALE * (beat - c.BASELINE_SCORE)) + 1
    model.add(objective >= floor - (SCALE * SCALE if hinted else 0))
    model.maximize(objective)

    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1  # CPU time of every thread counts
    budget = seconds
    for _ in range(MAX_CUT_ROUNDS):
        solver.parameters.max_time_in_seconds = budget
        status = solver.solve(model)
        budget -= solver.wall_time
        name = solver.status_name(status)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None, name
        chosen = {sides[k][0]: k for k in range(len(sides)) if solver.value(use[k])}
        first = next(iter(chosen))
        loop, at = [], first
        while True:
            loop.append(chosen[at])
            at = sides[chosen[at]][1]
            if at == first:
                break
        crossing = _crossings(loop, sides, points)
        if not crossing:
            break
        for k1, k2 in crossing:
            model.add_bool_or([~use[k1], ~use[k2]])
        if budget <= 0.5:
            return None, "OUT_OF_TIME"
    else:
        return None, "TOO_MANY_CUTS"

    pieces, connectors = build_pieces(
        [(sides[k][2], sides[k][3]) for k in loop], solver, counts
    )
    start = points[sides[loop[0]][0]]
    construction = Construction(
        lattice.to_world(*start),
        (lattice.heading + 45.0 * sides[loop[0]][2]) % 360.0,
        pieces,
        connectors,
    )
    score = c.BASELINE_SCORE + solver.objective_value / (SCALE * SCALE)
    corners = [tuple(points[sides[k][0]]) for k in loop]
    used = {lattice.layer[sides[k][0]] for k in loop} if lattice.layer else None
    found = Found(construction, score, status == cp_model.OPTIMAL, corners, used)
    return found, name


def _cell_area(model, lattice: MixedLattice, use):
    """Area (times SCALE) as the cells inside the loop, for a right-angled
    lattice. The distinct node coordinates cut the room into rectangles of
    different sizes; each is inside or outside, and across every edge between
    two of them the difference equals the sides running along that edge (inside
    on the left). This bounds the search far more tightly than the shoelace sum
    of the sides, which is what lets the solver find big loops quickly."""
    xs, ys = lattice.xs, lattice.ys
    at_x = {round(v * 1000): i for i, v in enumerate(xs)}
    at_y = {round(v * 1000): i for i, v in enumerate(ys)}
    cell = {
        (p, q): model.new_bool_var(f"c{p}_{q}")
        for p, q in zip(*np.nonzero(lattice.cells))
    }

    def inside(p, q):
        return cell.get((p, q), 0)

    h_edges: dict[tuple[int, int], list] = {}
    v_edges: dict[tuple[int, int], list] = {}
    for k, (a, b, d, _, _) in enumerate(lattice.sides):
        (ax, ay), (bx, by) = lattice.points[a], lattice.points[b]
        if d in (0, 4):  # east, west: along row y
            q = at_y[round(ay * 1000)]
            p1, p2 = sorted((at_x[round(ax * 1000)], at_x[round(bx * 1000)]))
            for p in range(p1, p2):
                h_edges.setdefault((p, q), []).append(use[k] if d == 0 else -use[k])
        else:  # north, south: along column x
            p = at_x[round(ax * 1000)]
            q1, q2 = sorted((at_y[round(ay * 1000)], at_y[round(by * 1000)]))
            for q in range(q1, q2):
                v_edges.setdefault((p, q), []).append(use[k] if d == 2 else -use[k])
    nx, ny = len(xs), len(ys)
    for p in range(nx - 1):
        for q in range(ny):
            if (p, q) in cell or (p, q - 1) in cell:
                terms = h_edges.get((p, q), [])
                model.add(inside(p, q) - inside(p, q - 1) == sum(terms))
    for p in range(nx):
        for q in range(ny - 1):
            if (p - 1, q) in cell or (p, q) in cell:
                terms = v_edges.get((p, q), [])
                model.add(inside(p - 1, q) - inside(p, q) == sum(terms))
    return sum(
        round(SCALE * (xs[p + 1] - xs[p]) * (ys[q + 1] - ys[q])) * v
        for (p, q), v in cell.items()
    )


def _add_hint(model, use, sides, points, corners) -> bool:
    """Hint the loop through `corners` (lattice frame, in order) if each of its
    sides is a side of the lattice. Returns whether it was hinted."""
    where = {(round(x, 3), round(y, 3)): n for n, (x, y) in enumerate(points)}
    nodes = [where.get((round(x, 3), round(y, 3))) for x, y in corners]
    if any(n is None for n in nodes):
        return False
    index = {(s[0], s[1]): k for k, s in enumerate(sides)}
    chosen = [index.get(pair) for pair in zip(nodes, nodes[1:] + nodes[:1])]
    if any(k is None for k in chosen):
        return False
    picked = set(chosen)
    for k, v in enumerate(use):
        model.add_hint(v, k in picked)
    return True


def _crossings(loop: list[int], sides, points) -> list[tuple[int, int]]:
    """Pairs of sides of the loop that touch or cross, other than neighbours
    sharing their corner."""
    segments = [LineString([points[sides[k][0]], points[sides[k][1]]]) for k in loop]
    n = len(loop)
    out = []
    for p in range(n):
        for q in range(p + 2, n):
            if p == 0 and q == n - 1:
                continue
            if segments[p].distance(segments[q]) < c.TOL:
                out.append((loop[p], loop[q]))
    return out
