"""Polishing a finished loop off any grid: slide parts of it while that pays.

Every search stage places corners on some lattice, so the best loop it finds can
sit a unit or two short of where it should: 10 tall in a 12-tall hallway because
12 is not on a 5-unit grid. A slide fixes that directly. Pick two sides that run
in the same direction and move everything between them along that direction by
a whole number of units: those two sides grow or shrink by that much, every side
between them keeps its length and direction, and the loop stays closed. Positions
are no longer tied to any grid.

A slide is kept when the inventory can still build every side, the loop stays
simple and inside the room, and the simulator scores it higher. Slides repeat
until none helps or time runs out.
"""

import math
import time

from shapely.geometry import Polygon

import src.constants as c
from players.player10.assign import Assigner
from src.enclosure import Construction, score_construction, validate_construction
from src.pieces import Connector, ConnectorType, Piece, PieceType

MAX_SLIDE = 8  # units
LENGTH_TOL = 1e-6


def corners_of(vertices: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The corners of a loop given by its piece joints: drop straight joints."""
    out = []
    n = len(vertices)
    for i, (x, y) in enumerate(vertices):
        ax, ay = vertices[i - 1]
        bx, by = vertices[(i + 1) % n]
        if abs((x - ax) * (by - y) - (y - ay) * (bx - x)) > 1e-6:
            out.append((x, y))
    return out


def _sides(corners):
    """(length, unit direction) of each side, corner i to corner i + 1; None if a
    side has shrunk to nothing."""
    out = []
    for i, (x, y) in enumerate(corners):
        nx, ny = corners[(i + 1) % len(corners)]
        length = math.hypot(nx - x, ny - y)
        if length < c.MIN_WALL_LENGTH - LENGTH_TOL:
            return None
        out.append((length, ((nx - x) / length, (ny - y) / length)))
    return out


def construction_for(corners, assigner: Assigner) -> Construction | None:
    """Pieces for a counterclockwise loop of whole-length sides, or None."""
    sides = _sides(corners)
    if sides is None:
        return None
    lengths = [round(length) for length, _ in sides]
    if any(abs(L - length) > LENGTH_TOL for L, (length, _) in zip(lengths, sides)):
        return None
    if any(L > c.MAX_FACE_LENGTH or L < c.MIN_WALL_LENGTH for L in lengths):
        return None
    assignment = assigner.assign(tuple(lengths))
    if assignment is None:
        return None
    pieces, connectors = [], []
    for i, keys in enumerate(assignment.faces):
        (ax, ay), (bx, by) = sides[i - 1][1], sides[i][1]
        turn = math.degrees(math.atan2(ax * by - ay * bx, ax * bx + ay * by))
        kind = (
            ConnectorType.DIAGONAL if abs(abs(turn) - 45) < 1 else ConnectorType.RIGHT
        )
        for q, (is_gate, length) in enumerate(keys):
            pieces.append(Piece(PieceType.GATE if is_gate else PieceType.WALL, length))
            connectors.append(
                Connector(kind, reflex=turn < 0)
                if q == 0
                else Connector(ConnectorType.STRAIGHT)
            )
    dx, dy = sides[0][1]
    heading = math.degrees(math.atan2(dy, dx)) % 360.0
    return Construction(corners[0], heading, pieces, connectors)


def _estimate(corners, weights, bonus: float) -> float:
    polygon = Polygon(corners)
    return (
        c.BASELINE_SCORE + weights.A * polygon.area + weights.C * polygon.length + bonus
    )


def polish(construction, room, inventory, weights, assigner: Assigner, deadline: float):
    """The best loop reachable from `construction` by slides, with its score."""
    result = validate_construction(construction, room, inventory)
    if not result.valid:
        return construction, None
    best, best_score = construction, score_construction(result, weights)
    # slides keep the gates where they are, so the bonus stays as it is
    bonus = weights.G if result.second_gate_bonus else 0.0
    corners = corners_of(result.vertices)
    if Polygon(corners).exterior.is_ccw is False:
        corners = corners[::-1]
    improved = True
    while improved and time.process_time() < deadline:
        improved = False
        sides = _sides(corners)
        n = len(corners)
        moves = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                di, dj = sides[i][1], sides[j][1]
                if abs(di[0] * dj[1] - di[1] * dj[0]) > 1e-9:
                    continue  # not parallel
                # move corners i+1 .. j (the chain after side i up to side j)
                for step in range(-MAX_SLIDE, MAX_SLIDE + 1):
                    if step:
                        moves.append((i, j, step))
        candidates = []
        for i, j, step in moves:
            if time.process_time() > deadline:
                break  # a long loop has many slides; never run past the budget
            ux, uy = sides[i][1]
            moved = list(corners)
            k = (i + 1) % n
            while True:
                x, y = corners[k]
                moved[k] = (x + step * ux, y + step * uy)
                if k == j:
                    break
                k = (k + 1) % n
            if _sides(moved) is None:
                continue
            polygon = Polygon(moved)
            if not polygon.is_valid or polygon.area <= 0:
                continue
            estimate = _estimate(moved, weights, bonus)
            if estimate > best_score + 1e-6:
                candidates.append((estimate, moved))
        candidates.sort(key=lambda e: -e[0])
        for _, moved in candidates:
            if time.process_time() > deadline:
                break
            if not room.polygon.buffer(c.TOL).contains(Polygon(moved)):
                continue
            built = construction_for(moved, assigner)
            if built is None:
                continue
            result = validate_construction(built, room, inventory)
            if not result.valid:
                continue
            score = score_construction(result, weights)
            if score > best_score + 1e-6:
                best, best_score, corners = built, score, moved
                improved = True
                break
    return best, best_score


# ------------------------------------------------------------ diagonal pairs
#
# A 45-degree side of whole length L moves L/sqrt(2) along each axis, never a
# whole number, so one diagonal alone can never close a loop of whole-length
# sides. Two opposite ones can: put a diagonal of length L at corner p and the
# opposite diagonal at corner q, and the part of the loop between them shifts
# by that diagonal while every other side keeps its length. Shortening the two
# sides at p and the two matching sides at q by the same amounts keeps the
# loop closed too, so a pair can also cut both corners off (the staircase case).

MAX_CUT = 3  # extra units the sides next to a diagonal may shrink or grow by
SPLIT_NEAR = 4  # a diagonal partway along a side goes this close to its ends
MAX_SHIFT_CORNERS = 40
SHIFT_SECONDS = 2.0


def _frame(corners) -> float | None:
    """The loop's grid heading (degrees mod 90), or None if a side is off it."""
    n = len(corners)
    k = max(range(n), key=lambda k: math.dist(corners[k], corners[(k + 1) % n]))
    (ax, ay), (bx, by) = corners[k], corners[(k + 1) % n]
    heading = math.degrees(math.atan2(by - ay, bx - ax)) % 90.0
    for t in range(n):
        (ax, ay), (bx, by) = corners[t], corners[(t + 1) % n]
        a = (math.degrees(math.atan2(by - ay, bx - ax)) - heading) % 45.0
        if min(a, 45.0 - a) > 1e-3:
            return None
    return heading


def _walk(start, heading, sides):
    """Corners of the loop that starts at `start` with (dir8, length) sides."""
    out, (x, y) = [], start
    for d, length in sides:
        out.append((x, y))
        t = math.radians(heading + 45.0 * d)
        x, y = x + length * math.cos(t), y + length * math.sin(t)
    return out


def chamfer_pairs(construction, room, inventory, weights, assigner, deadline):
    """The best loop one pair of opposite diagonals makes, if it beats this one."""
    result = validate_construction(construction, room, inventory)
    if not result.valid:
        return construction, None
    base = score_construction(result, weights)
    if inventory.connectors.get(ConnectorType.DIAGONAL, 0) < 4:
        return construction, None  # each pair turns two right corners into four 45s
    corners = corners_of(result.vertices)
    if Polygon(corners).exterior.is_ccw is False:
        corners = corners[::-1]
    heading = _frame(corners)
    if heading is None:
        return construction, None
    n = len(corners)
    sides = []
    for k in range(n):
        (ax, ay), (bx, by) = corners[k], corners[(k + 1) % n]
        d = round((math.degrees(math.atan2(by - ay, bx - ax)) - heading) / 45.0) % 8
        sides.append((d, round(math.hypot(bx - ax, by - ay))))
    bonus = weights.G if result.second_gate_bonus else 0.0
    used = sum(1 for k in range(n) if (sides[k][0] - sides[k - 1][0]) % 8 in (1, 7))
    if inventory.connectors.get(ConnectorType.DIAGONAL, 0) < used + 4:
        return construction, None
    # where a diagonal can go: at a 90-degree corner k (between side k-1 and
    # side k), or partway along side k, t units from its start, leaving it at
    # 45 degrees either way and coming back to the side's own direction
    spots = []  # (side k, t, diagonal direction)
    for k in range(n):
        d, length = sides[k]
        turn = (d - sides[k - 1][0] + 4) % 8 - 4
        if abs(turn) == 2:
            spots.append((k, 0, (sides[k - 1][0] + turn // 2) % 8))
        for t in sorted({*range(1, min(SPLIT_NEAR, length)), *range(max(1, length - SPLIT_NEAR + 1), length)}):
            if c.MIN_WALL_LENGTH <= t <= length - c.MIN_WALL_LENGTH:
                spots.append((k, t, (d + 1) % 8))
                spots.append((k, t, (d - 1) % 8))
    diagonals = [L for L in assigner.lengths if L <= c.MAX_FACE_LENGTH]
    container = room.polygon.buffer(c.TOL)
    candidates = []
    for x, (p, tp, mid) in enumerate(spots):
        if time.process_time() > deadline:
            break
        for q, tq, other in spots[x + 1 :]:
            if other != (mid + 4) % 8 or q == p:
                continue
            for L in diagonals:
                r = int(L / math.sqrt(2))
                if tp == 0 and tq == 0:
                    cuts = sorted({0, *range(max(0, r - MAX_CUT), r + MAX_CUT + 1)})
                    options = [(a, b) for a in cuts for b in cuts]
                else:
                    options = [(0, 0)]  # a split side needs no cut: the chain shifts
                for a, b in options:
                    if tp == 0 and tq == 0:
                        new = _pair(sides, p, q, mid, L, a, b)
                    else:
                        new = _split_pair(sides, (p, tp, mid), (q, tq, other), L)
                    if new is None:
                        continue
                    moved = _walk(corners[0], heading, new)
                    polygon = Polygon(moved)
                    if not polygon.is_valid or polygon.area <= 0:
                        continue
                    estimate = _estimate(moved, weights, bonus)
                    if estimate > base + 1e-6:
                        candidates.append((estimate, moved))
    candidates.sort(key=lambda e: -e[0])
    for _, moved in candidates:
        if time.process_time() > deadline:
            break
        if not container.contains(Polygon(moved)):
            continue
        built = construction_for(moved, assigner)
        if built is None:
            continue
        check = validate_construction(built, room, inventory)
        if not check.valid:
            continue
        score = score_construction(check, weights)
        if score > base + 1e-6:
            return built, score
    return construction, None


def _pair(sides, p, q, mid, L, a, b):
    """`sides` with a diagonal of length L (direction `mid`) at corner p and the
    opposite one at corner q. At p the side before shrinks by a and the side
    after by b; at q the sides running the opposite ways shrink by the same, so
    the loop still closes. None if a side would vanish or grow too long."""
    n = len(sides)
    new = [list(s) for s in sides]
    before_p, after_p = (p - 1) % n, p
    before_q, after_q = (q - 1) % n, q
    # at q the side before runs opposite to the one after p, and vice versa
    if sides[before_q][0] == (sides[before_p][0] + 4) % 8:
        pairs = ((before_p, a), (after_p, b), (before_q, a), (after_q, b))
    else:  # q turns the other way round: its sides pair up the other way
        pairs = ((before_p, a), (after_p, b), (before_q, b), (after_q, a))
    for k, cut in pairs:
        new[k][1] -= cut
    for d, length in new:
        if length < c.MIN_WALL_LENGTH or length > c.MAX_FACE_LENGTH:
            return None
    out = []
    for k in range(n):
        if k == p:
            out.append((mid, L))
        elif k == q:
            out.append(((mid + 4) % 8, L))
        out.append(tuple(new[k]))
    return out


def _split_pair(sides, first, second, L):
    """`sides` with diagonals of length L at two spots, at least one partway
    along a side (see chamfer_pairs). A corner spot keeps its sides as they are."""
    n = len(sides)
    by_side = {}
    for k, t, d in (first, second):
        by_side.setdefault(k, []).append((t, d))
    out = []
    for k in range(n):
        d, length = sides[k]
        cuts = sorted(by_side.get(k, []))
        at = 0
        for t, diagonal in cuts:
            if t == 0:
                out.append((diagonal, L))
                continue
            out.append((d, t - at))
            out.append((diagonal, L))
            at = t
        out.append((d, length - at))
    for d, length in out:
        if length < c.MIN_WALL_LENGTH or length > c.MAX_FACE_LENGTH:
            return None
    # two diagonals in a row, or a side that turns back on itself, is no loop
    for k in range(len(out)):
        turn = (out[k][0] - out[k - 1][0]) % 8
        if turn in (0, 3, 4, 5):
            return None
    return out


def shift_to_walls(construction, room, inventory, weights, assigner, deadline):
    """Slide the whole loop a fraction of a unit so a corner meets a wall, then
    slide its parts (polish). Sides have whole lengths, so every corner of a loop
    sits on one grid; this picks that grid's offset from the walls themselves."""
    result = validate_construction(construction, room, inventory)
    if not result.valid:
        return construction, None
    base = score_construction(result, weights)
    corners = corners_of(result.vertices)
    heading = _frame(corners)
    if heading is None or len(corners) > MAX_SHIFT_CORNERS:
        return construction, None
    t = math.radians(heading)
    axes = ((math.cos(t), math.sin(t)), (-math.sin(t), math.cos(t)))
    walls = list(zip(room.polygon.exterior.coords, list(room.polygon.exterior.coords)[1:]))
    shifts = set()
    for (x, y) in corners:
        for ux, uy in axes:
            for (ax, ay), (bx, by) in walls:
                # x + s*u on the wall a + w*(b - a): solve for s
                ex, ey = bx - ax, by - ay
                det = ux * (-ey) - uy * (-ex)
                if abs(det) < 1e-9:
                    continue
                rx, ry = ax - x, ay - y
                s_ = (rx * (-ey) - ry * (-ex)) / det
                w_ = (ux * ry - uy * rx) / det
                if 0 <= w_ <= 1 and 1e-6 < abs(s_) < 1.0:
                    shifts.add((round(s_ * ux, 6), round(s_ * uy, 6)))
    best, best_score = construction, base
    for dx, dy in sorted(shifts, key=lambda v: abs(v[0]) + abs(v[1])):
        if time.process_time() > deadline:
            break
        moved = construction_for([(x + dx, y + dy) for x, y in _ccw(corners)], assigner)
        if moved is None:
            continue
        if not validate_construction(moved, room, inventory).valid:
            continue
        slice_end = min(deadline, time.process_time() + SHIFT_SECONDS)
        polished, score = polish(moved, room, inventory, weights, assigner, slice_end)
        if score is not None and score > best_score + 1e-6:
            best, best_score = polished, score
    if best is construction:
        return construction, None
    return best, best_score


def _ccw(corners):
    return corners if Polygon(corners).exterior.is_ccw else corners[::-1]
