"""Right-angled loops on an adaptive lattice: fine along walls and narrow
passages, coarse everywhere else.

A plain lattice is either too coarse to reach the corners good loops use -- a
corridor 4 units tall, a side that ends 2 units from a wall -- or too fine to
solve in time. The best loops hug the room: their corners sit on walls, or on
the lines that run through narrow passages, and the open floor in between only
needs a coarse grid. So nodes go

- every unit along each wall that lies along a lattice axis, the line carried
  on across the room until it meets the far wall (a corner touching a wall can
  sit anywhere along it);
- every unit along every line across a narrow passage (two parallel walls at
  most NARROW apart), carried on the same way, so a loop can leave the passage
  and keep going straight into the rooms on either side;
- on a COARSE grid over the rest.

Sides join two nodes on the same row or column at a buildable length, as on the
plain lattice. The nodes no longer form cells, so the loop is solved by the same
model as the layered lattice (mixed.py): shoelace area, and crossings forbidden
as they turn up.
"""

import math
from itertools import combinations

import numpy as np
import shapely

import src.constants as c
from players.player10.circuit import local_room
from players.player10.mixed import MixedLattice

COARSE = 5
NARROW = 10  # passages at most this wide get a line every unit across them
AXIS_TOL = 1e-6
EAST, NORTH, WEST, SOUTH = 0, 2, 4, 6  # in eighths of a turn


def _key(x: float, y: float) -> tuple[int, int]:
    return (round(x * 1000), round(y * 1000))


def _fine_lines(local) -> tuple[set[float], set[float]]:
    """y of horizontal and x of vertical fine lines: walls along the axes, and
    every unit across passages at most NARROW wide."""
    corners = np.array(local.exterior.coords)[:-1]
    rows, cols = set(), set()
    h_walls, v_walls = [], []
    for k in range(len(corners)):
        (ax, ay), (bx, by) = corners[k], corners[(k + 1) % len(corners)]
        if abs(ay - by) < AXIS_TOL:
            rows.add(round(float(ay), 6))
            h_walls.append((min(ax, bx), max(ax, bx), float(ay)))
        elif abs(ax - bx) < AXIS_TOL:
            cols.add(round(float(ax), 6))
            v_walls.append((min(ay, by), max(ay, by), float(ax)))
    for walls, lines, horizontal in ((h_walls, rows, True), (v_walls, cols, False)):
        for (a1, b1, p), (a2, b2, q) in combinations(walls, 2):
            lo, hi = sorted((p, q))
            overlap = (max(a1, a2), min(b1, b2))
            if not (0 < hi - lo <= NARROW and overlap[1] - overlap[0] > AXIS_TOL):
                continue
            middle = ((overlap[0] + overlap[1]) / 2, (lo + hi) / 2)
            if not horizontal:
                middle = middle[::-1]
            if not local.contains(shapely.Point(middle)):
                continue  # the two walls face away from each other
            for v in range(math.ceil(lo), math.floor(hi) + 1):
                lines.add(float(v))
    return rows, cols


def build_adaptive_lattice(
    room_polygon,
    heading: float,
    origin,
    lengths: list[int],
    max_nodes: int,
    window: tuple[float, float, float, float] | None = None,
) -> MixedLattice | None:
    """The adaptive lattice in a frame turned by `heading` and anchored at
    `origin` (a room corner, so walls fall on whole coordinates), or None when it
    has more than `max_nodes` nodes. With a `window` (a box in that frame) only
    the biggest piece of the room inside the box is covered."""
    local = local_room(room_polygon, heading, origin)
    if window is not None:
        piece = local.intersection(shapely.box(*window))
        parts = getattr(piece, "geoms", [piece])
        local = max(parts, key=lambda g: g.area if g.geom_type == "Polygon" else 0)
        if local.geom_type != "Polygon" or local.area <= 0:
            return None
    region = local.buffer(c.TOL)
    shapely.prepare(region)
    x0, y0, x1, y1 = local.bounds
    rows, cols = _fine_lines(local)

    points: dict[tuple[int, int], tuple[float, float]] = {}

    def add(xs, ys):
        xs, ys = np.broadcast_arrays(np.asarray(xs, float), np.asarray(ys, float))
        inside = shapely.covers(region, shapely.points(xs, ys))
        for x, y in zip(xs[inside], ys[inside]):
            points.setdefault(_key(x, y), (float(x), float(y)))

    whole_x = np.arange(math.ceil(x0), math.floor(x1) + 1)
    whole_y = np.arange(math.ceil(y0), math.floor(y1) + 1)
    for y in rows:
        add(whole_x, y)
    for x in cols:
        add(x, whole_y)
    gx, gy = np.meshgrid(
        np.arange(math.ceil(x0 / COARSE), math.floor(x1 / COARSE) + 1) * COARSE,
        np.arange(math.ceil(y0 / COARSE), math.floor(y1 / COARSE) + 1) * COARSE,
        indexing="ij",
    )
    add(gx.ravel(), gy.ravel())
    if len(points) > max_nodes:
        return None

    keys = list(points)
    xy = np.array([points[k] for k in keys])
    by_row: dict[int, list[int]] = {}
    by_col: dict[int, list[int]] = {}
    for n, (kx, ky) in enumerate(keys):
        by_row.setdefault(ky, []).append(n)
        by_col.setdefault(kx, []).append(n)

    candidates = []
    for line, along, forward, backward in (
        (by_row, 0, EAST, WEST),
        (by_col, 1, NORTH, SOUTH),
    ):
        for members in line.values():
            members.sort(key=lambda n: xy[n][along])
            position = {round(xy[n][along] * 1000): i for i, n in enumerate(members)}
            for i, n in enumerate(members):
                here = xy[n][along]
                for length in lengths:
                    j = position.get(round((here + length) * 1000))
                    if j is None:
                        continue
                    end = members[j]
                    through = members[i + 1 : j]
                    candidates.append((n, end, forward, length, through))
                    candidates.append((end, n, backward, length, through))
    sides = []
    if candidates:
        segments = shapely.linestrings([[xy[s[0]], xy[s[1]]] for s in candidates])
        fits = shapely.contains(region, segments)
        sides = [s for s, ok in zip(candidates, fits) if ok]
    # the rectangles between neighbouring node coordinates, for the cell area
    xs = np.unique(np.round(xy[:, 0], 6))
    ys = np.unique(np.round(xy[:, 1], 6))
    cx, cy = np.meshgrid(xs, ys, indexing="ij")
    cells = shapely.contains(
        region, shapely.box(cx[:-1, :-1], cy[:-1, :-1], cx[1:, 1:], cy[1:, 1:])
    )
    return MixedLattice(
        heading, COARSE, tuple(origin), xy, sides, xs=xs, ys=ys, cells=cells
    )


def windows(room_polygon, heading: float, origin, size: float):
    """Square boxes of side `size`, overlapping by half, covering the room in the
    lattice frame: for rooms too big to cover with one adaptive lattice."""
    x0, y0, x1, y1 = local_room(room_polygon, heading, origin).bounds
    step = size / 2
    out = []
    for x in np.arange(x0, max(x0 + step, x1 - step), step):
        for y in np.arange(y0, max(y0 + step, y1 - step), step):
            out.append(
                (float(x), float(y), float(min(x + size, x1)), float(min(y + size, y1)))
            )
    return out
