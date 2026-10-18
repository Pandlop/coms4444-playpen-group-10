"""Rebuilding stretches of a finished loop on fine grids laid only outside it.

A loop found on a coarse lattice leaves gaps between itself and the walls. A fine
lattice over the whole room is too big to solve, but most of the loop is already
right: only the stretches next to a gap need to move. So, for each gap:

- take a short run of the loop that borders it, corner A to corner B;
- lay a fine lattice (adaptive, or layered for 45-degree sides) over the gap plus
  a band around that run, in the loop's own frame with A at the origin;
- let CP-SAT pick the best path from A to B through that lattice. The rest of the
  loop is fixed: its pieces, connectors, gates, area and perimeter are constants,
  and the path may not touch it except at A and B.

The path's pieces come straight from the model, so the inventory check is exact
and no assignment search over the whole loop is needed. A new loop is kept only
when the simulator validates it and scores it strictly higher.
"""

import math
import time

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon

import src.constants as c
from players.player10 import adaptive, mixed
from players.player10.assign import Assigner
from players.player10.circuit import SCALE, cp_model, local_room
from src.enclosure import Construction, score_construction, validate_construction
from src.pieces import Connector, ConnectorType, Piece, PieceType

FORBIDDEN = mixed.FORBIDDEN_TURNS
WINDOW = 10  # corners freed at a time
STEP = 5  # windows overlap by half
NEAR = 12  # corners this close to a gap border it
BAND = 12  # the path may also move this far either side of the freed run
REACH = 25  # how far into the gap the lattice goes
MIN_GAP = 20.0
SOLVE_SECONDS = 5.0
MAX_NODES = 1500
LAYERED_SPACINGS = (1, 2, 5)
MAX_CUT_ROUNDS = 10


def _dir(a, b) -> int:
    return round(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) / 45.0) % 8


class Refiner:
    def __init__(self, room, inventory, weights, assigner: Assigner):
        self.room, self.inventory, self.weights = room, inventory, weights
        self.assigner = assigner
        self.tried: set = set()  # subproblems that gave nothing

    # ------------------------------------------------------------------ driver
    def run(self, construction, kind: str, deadline: float):
        """Improve `construction` with `kind` ("adaptive" or "layered") lattices
        until nothing helps or the deadline passes. Returns (best, score)."""
        result = validate_construction(construction, self.room, self.inventory)
        if not result.valid:
            return construction, None
        best, best_score = construction, score_construction(result, self.weights)
        improved = True
        while improved and time.process_time() < deadline - 1.0:
            improved = False
            for chain_ij, gap, frame in self._windows(best):
                if time.process_time() > deadline - 1.0:
                    break
                out = self._refine(best, best_score, gap, chain_ij, frame, kind, deadline)
                if out is not None:
                    best, best_score = out
                    improved = True
                    break
        return best, best_score

    def _loop(self, construction):
        """Corners (world), faces' pieces, and the lattice frame (heading, A-free)."""
        res = validate_construction(construction, self.room, self.inventory)
        faces = []
        for i, (piece, conn) in enumerate(
            zip(construction.pieces, construction.connectors)
        ):
            if i == 0 or conn.connector_type != ConnectorType.STRAIGHT:
                faces.append((i, []))
            faces[-1][1].append(piece)
        corners = [tuple(map(float, res.vertices[f[0]])) for f in faces]
        return corners, [f[1] for f in faces]

    def _windows(self, construction):
        corners, _ = self._loop(construction)
        n = len(corners)
        loop = Polygon(corners)
        if n < 4 or not loop.is_valid:
            return
        # frame: the longest side's direction, modulo a right angle
        k = max(range(n), key=lambda k: math.dist(corners[k], corners[(k + 1) % n]))
        (ax, ay), (bx, by) = corners[k], corners[(k + 1) % n]
        heading = math.degrees(math.atan2(by - ay, bx - ax)) % 90.0
        for t in range(n):
            (ax, ay), (bx, by) = corners[t], corners[(t + 1) % n]
            a = (math.degrees(math.atan2(by - ay, bx - ax)) - heading) % 45.0
            if min(a, 45.0 - a) > 1e-3:
                return  # a side off the 45-degree directions: no lattice fits it
        gaps = self.room.polygon.difference(loop)
        gaps = sorted(
            (g for g in getattr(gaps, "geoms", [gaps]) if g.area > MIN_GAP),
            key=lambda g: -g.area,
        )
        for gap in gaps:
            near = [k for k in range(n) if gap.distance(Point(corners[k])) < NEAR]
            if len(near) < 2:
                continue
            runs, run = [], [near[0]]
            for k in near[1:]:
                if k == run[-1] + 1:
                    run.append(k)
                else:
                    runs.append(run)
                    run = [k]
            runs.append(run)
            if len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == n - 1:
                runs[0] = runs.pop() + runs[0]  # the run wraps past corner 0
            for run in runs:
                for start in range(0, max(1, len(run) - 2), STEP):
                    chunk = run[start : start + WINDOW]
                    yield ((chunk[0] - 1) % n, (chunk[-1] + 1) % n), gap, heading

    # ------------------------------------------------------------ one window
    def _refine(self, construction, base_score, gap, chain_ij, heading, kind, deadline):
        W, inv = self.weights, self.inventory
        world, faces = self._loop(construction)
        n = len(world)
        i, j = chain_ij
        chain = [(i + t) % n for t in range(((j - i) % n) + 1)]  # A .. B
        if len(chain) < 3 or len(chain) >= n - 1:
            return None
        A = world[i]
        key = (kind, tuple(world[k] for k in chain), world[(i - 1) % n], world[(j + 1) % n])
        if key in self.tried:
            return None
        self.tried.add(key)

        # everything in the lattice frame: A at the origin, sides on the axes
        def local(p):
            x, y = p[0] - A[0], p[1] - A[1]
            t = math.radians(-heading)
            return (
                x * math.cos(t) - y * math.sin(t),
                x * math.sin(t) + y * math.cos(t),
            )

        corners = [local(p) for p in world]
        corners = [(round(x, 6), round(y, 6)) for x, y in corners]
        sign = 1 if Polygon(corners).exterior.is_ccw else -1  # area sign of this walk
        room = local_room(self.room.polygon, heading, A)
        gap = affinity.rotate(
            affinity.translate(gap, -A[0], -A[1]), -heading, origin=(0, 0)
        )
        freed = chain[:-1]
        fixed = [k for k in range(n) if k not in freed]
        B = corners[j]
        d_in_A = _dir(corners[(i - 1) % n], corners[i])
        d_out_B = _dir(B, corners[(j + 1) % n])

        # what the fixed part leaves for the path
        left = inv.copy()
        straights_used, fixed_gated = 0, 0
        for k in fixed:
            straights_used += len(faces[k]) - 1
            fixed_gated += any(p.piece_type == PieceType.GATE for p in faces[k])
            for p in faces[k]:
                pool = left.gates if p.piece_type == PieceType.GATE else left.walls
                pool[int(p.length)] -= 1
        left.connectors[ConnectorType.STRAIGHT] = (
            left.connectors.get(ConnectorType.STRAIGHT, 0) - straights_used
        )

        def diagonal_turn(k):
            a, b, cc = corners[(k - 1) % n], corners[k], corners[(k + 1) % n]
            return (_dir(b, cc) - _dir(a, b)) % 8 in (1, 7)

        fixed_corners = [k for k in range(n) if k not in chain]
        rights_left = inv.connectors.get(ConnectorType.RIGHT, 0) - sum(
            not diagonal_turn(k) for k in fixed_corners
        )
        diags_left = inv.connectors.get(ConnectorType.DIAGONAL, 0) - sum(
            diagonal_turn(k) for k in fixed_corners
        )
        sub = Assigner(left, want_second_gate=W.G > 0)

        run_line = LineString([corners[k] for k in chain])
        region = (
            gap.intersection(run_line.buffer(REACH))
            .buffer(0.5)
            .union(run_line.buffer(BAND))
            .intersection(room.buffer(c.TOL))
        )
        parts = getattr(region, "geoms", [region])
        region = next((g for g in parts if g.covers(Point(0, 0))), None)
        if region is None or region.geom_type != "Polygon":
            return None
        fixed_line = LineString([corners[(j + t) % n] for t in range(((i - j) % n) + 1)])
        lengths = self.assigner.lengths

        if kind == "layered":
            short = [L for L in lengths if L <= 10] or lengths[:2]
            lattice = None
            for spacing in LAYERED_SPACINGS:
                lattice = mixed.build_mixed_lattice(
                    region, 0.0, spacing, (0.0, 0.0), lengths, short, 1, MAX_NODES
                )
                if lattice is not None:
                    break
        else:
            x0, y0, x1, y1 = region.bounds
            lattice = adaptive.build_adaptive_lattice(
                room, 0.0, (0.0, 0.0), lengths, 10**6, (x0 - 1, y0 - 1, x1 + 1, y1 + 1)
            )
        if lattice is None:
            return None
        pts_all = np.asarray(lattice.points, float)
        if len(pts_all) == 0:
            return None
        points = shapely.points(pts_all)
        keep = shapely.covers(region.buffer(c.TOL), points)
        keep &= shapely.distance(fixed_line, points) > 1e-6
        dA = ((pts_all - np.array([0.0, 0.0])) ** 2).sum(1)
        dB = ((pts_all - np.array(B)) ** 2).sum(1)
        keyA, keyB = int(dA.argmin()), int(dB.argmin())
        if dA[keyA] > 1e-9 or dB[keyB] > 1e-9:
            return None  # A or B is not on this lattice
        keep[keyA] = keep[keyB] = True
        ids = {old: new for new, old in enumerate(map(int, keep.nonzero()[0]))}
        pts = pts_all[keep]
        if len(pts) > MAX_NODES:
            return None
        nA, nB = ids[keyA], ids[keyB]
        sides = []
        for a, b, d, L, through in lattice.sides:
            if a not in ids or b not in ids or b == keyA or a == keyB:
                continue
            if a == keyA and (d - d_in_A) % 8 in FORBIDDEN:
                continue
            if b == keyB and (d_out_B - d) % 8 in FORBIDDEN:
                continue
            sides.append((ids[a], ids[b], d, L, [ids[t] for t in through if t in ids]))
        if sides:
            segs = shapely.linestrings([[pts[s[0]], pts[s[1]]] for s in sides])
            hits = shapely.intersects(segs, fixed_line)
            ok = []
            for s, seg, hit in zip(sides, segs, hits):
                if hit:
                    rest = seg.intersection(fixed_line)
                    if s[0] == nA:
                        rest = rest.difference(Point(0, 0).buffer(1e-6))
                    if s[1] == nB:
                        rest = rest.difference(Point(B).buffer(1e-6))
                    if not rest.is_empty:
                        continue
                ok.append(s)
            sides = ok
        if not sides:
            return None

        m = cp_model.CpModel()
        use = [m.new_bool_var("") for _ in sides]
        arcs = [(s[0], s[1], use[k]) for k, s in enumerate(sides)]
        arcs.append((nB, nA, m.new_constant(1)))
        arcs += [(v, v, m.new_bool_var("")) for v in range(len(pts)) if v not in (nA, nB)]
        m.add_circuit(arcs)
        N = len(pts)
        touch = [[] for _ in range(N)]
        arrive = [[[] for _ in range(8)] for _ in range(N)]
        leave = [[[] for _ in range(8)] for _ in range(N)]
        for k, (a, b, d, L, through) in enumerate(sides):
            touch[a].append(use[k])
            for t in through:
                touch[t].append(use[k])
            leave[a][d].append(use[k])
            arrive[b][d].append(use[k])
        diag_turns = []
        for v in range(N):
            if len(touch[v]) > 1:
                m.add_at_most_one(touch[v])
            if v in (nA, nB):
                continue
            for x in range(8):
                if not arrive[v][x]:
                    continue
                for y in range(8):
                    if leave[v][y] and (y - x) % 8 in FORBIDDEN:
                        m.add(sum(arrive[v][x]) + sum(leave[v][y]) <= 1)
            into = [u for d in (1, 3, 5, 7) for u in arrive[v][d]]
            out = [u for d in (1, 3, 5, 7) for u in leave[v][d]]
            if not into and not out:
                continue
            t = m.new_bool_var("")
            si, so = sum(into), sum(out)
            m.add(t >= si - so)
            m.add(t >= so - si)
            m.add(t <= si + so)
            m.add(t <= 2 - si - so)
            diag_turns.append(t)
        diag_turns += [
            use[k] for k, s in enumerate(sides) if s[0] == nA and s[2] % 2 != d_in_A % 2
        ]
        diag_turns += [
            use[k] for k, s in enumerate(sides) if s[1] == nB and s[2] % 2 != d_out_B % 2
        ]
        m.add(sum(diag_turns) <= diags_left)
        m.add(sum(use) + 1 - sum(diag_turns) <= rights_left)

        by_length: dict[int, list] = {}
        for k, s in enumerate(sides):
            by_length.setdefault(s[3], []).append(use[k])
        counts, piece_use = {}, {key: [] for key in sub.stock}
        straights, gated = [], []
        for L, chosen in by_length.items():
            if L not in sub.compositions:
                for u in chosen:
                    m.add(u == 0)
                continue
            split = []
            plain, with_gate = sub.compositions[L]
            for comp in plain + with_gate:
                v = m.new_int_var(0, len(chosen), "")
                counts[(L, comp)] = v
                split.append(v)
                for key_ in set(comp):
                    piece_use[key_].append(comp.count(key_) * v)
                if len(comp) > 1:
                    straights.append((len(comp) - 1) * v)
                if any(k_[0] for k_ in comp):
                    gated.append(v)
            m.add(sum(split) == sum(chosen))
        for key_, terms in piece_use.items():
            if terms:
                m.add(sum(terms) <= sub.stock[key_])
        m.add(sum(straights) <= max(0, sub.straights))
        m.add(sum(gated) >= max(0, 1 - fixed_gated))
        bonus = 0
        if W.G > 0:
            bonus = m.new_bool_var("")
            m.add(fixed_gated + sum(gated) >= 2).only_enforce_if(bonus)

        fixed_twice_area = sum(
            corners[k][0] * corners[(k + 1) % n][1]
            - corners[(k + 1) % n][0] * corners[k][1]
            for k in fixed
        )
        fixed_perimeter = sum(math.dist(corners[k], corners[(k + 1) % n]) for k in fixed)
        area = sign * sum(
            use[k]
            * round(
                SCALE
                * (pts[s[0]][0] * pts[s[1]][1] - pts[s[1]][0] * pts[s[0]][1])
                / 2
            )
            for k, s in enumerate(sides)
        ) + sign * round(SCALE * fixed_twice_area / 2)
        perimeter = sum(use[k] * s[3] for k, s in enumerate(sides))
        objective = (
            round(SCALE * W.A) * area
            + round(SCALE * SCALE * W.C) * perimeter
            + round(SCALE * SCALE * W.G) * bonus
        )
        constant = round(SCALE * SCALE * W.C * fixed_perimeter)
        m.add(
            objective + constant
            >= math.floor(SCALE * SCALE * (base_score - c.BASELINE_SCORE))
            - SCALE * SCALE
        )
        m.maximize(objective)
        # the current run as a hint, when it lies on this lattice
        where = {(round(float(p[0]), 3), round(float(p[1]), 3)): v for v, p in enumerate(pts)}
        index = {(s[0], s[1]): k for k, s in enumerate(sides)}
        now = [where.get((round(corners[k][0], 3), round(corners[k][1], 3))) for k in chain]
        if None not in now:
            picked = {index.get((now[t], now[t + 1])) for t in range(len(now) - 1)}
            if None not in picked:
                for k, u in enumerate(use):
                    m.add_hint(u, k in picked)

        solver = cp_model.CpSolver()
        solver.parameters.num_workers = 1
        path = None
        for _ in range(MAX_CUT_ROUNDS):
            seconds = min(SOLVE_SECONDS, deadline - time.process_time() - 0.5)
            if seconds < 0.5:
                return None
            solver.parameters.max_time_in_seconds = seconds
            status = solver.solve(m)
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                return None
            nxt = {sides[k][0]: k for k in range(len(sides)) if solver.value(use[k])}
            path, at = [], nA
            while at != nB:
                k = nxt[at]
                path.append(k)
                at = sides[k][1]
            segs = [LineString([pts[sides[k][0]], pts[sides[k][1]]]) for k in path]
            crossings = [
                (path[p], path[q])
                for p in range(len(path))
                for q in range(p + 2, len(path))
                if segs[p].distance(segs[q]) < c.TOL
            ]
            if not crossings:
                break
            for a, b in crossings:
                m.add_bool_or([~use[a], ~use[b]])
        else:
            return None

        pool: dict[int, list] = {}
        for (L, comp), v in counts.items():
            pool.setdefault(L, []).extend([comp] * solver.value(v))
        loop = [(sides[k][2], list(pool[sides[k][3]].pop())) for k in path]
        for t in range((i - j) % n):
            f = (j + t) % n
            loop.append(
                (
                    _dir(corners[f], corners[(f + 1) % n]),
                    [(p.piece_type == PieceType.GATE, int(p.length)) for p in faces[f]],
                )
            )
        pieces, connectors = [], []
        for q, (d, keys) in enumerate(loop):
            turn = (d - loop[q - 1][0] + 4) % 8 - 4
            corner = ConnectorType.DIAGONAL if abs(turn) == 1 else ConnectorType.RIGHT
            for r, (is_gate, L) in enumerate(keys):
                pieces.append(Piece(PieceType.GATE if is_gate else PieceType.WALL, L))
                connectors.append(
                    Connector(corner, reflex=turn * sign < 0)
                    if r == 0
                    else Connector(ConnectorType.STRAIGHT)
                )
        built = Construction(A, (heading + loop[0][0] * 45.0) % 360.0, pieces, connectors)
        result = validate_construction(built, self.room, inv)
        if not result.valid:
            return None
        score = score_construction(result, W)
        if score > base_score + 1e-6:
            return built, score
        return None
