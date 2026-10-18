import heapq
import math
import time

import src.constants as c
from players.player0 import Player
from players.player10 import adaptive, circuit, mixed, polish, refine
from players.player10.assign import Assigner, Assignment
from players.player10.placement import RoomSearch
from players.player10.shapes import (
    CORNERS_NEEDED,
    HEADING_STEP_DEG,
    L_SHAPE,
    MEASURES,
    HEXAGON,
    OCTAGON,
    RECTANGLE,
    SYMMETRY_DEG,
    U_SHAPE,
    Shape,
    contains_smaller,
    inner_radius,
    inner_rectangles,
    make_shape,
)
from src.enclosure import Construction, score_construction, validate_construction
from src.pieces import Connector, ConnectorType, Piece, PieceType

# CPU seconds the search may use. Nothing can interrupt a player on Windows, so
# the player has to stop itself well inside the simulator's limit.
TIME_BUDGET = 20.0
# how far into the budget each stage of the search may run
RECTANGLE_SHARE = 0.3
L_SHARE = 0.5
HEXAGON_SECONDS = 10.0  # hexagons, held back until the lattice stages are done
HELD_OCTAGON_SECONDS = 30.0  # octagons again, once a hexagon set a higher bar
OCTAGON_SHARE = 0.8
FALLBACK_BUDGET = 3.0
FALLBACK_SHAPES = 40
RECENT_FAILURES = 400
SHRINK_STEPS = 4

# the lattice stages run after the templates, each until its budget of CPU
# seconds after the start -- well inside the 300 s limit: right-angled loops on
# plain lattices (circuit.py), then on adaptive lattices, fine along walls and
# narrow passages (adaptive.py), then loops mixing 45- and 90-degree corners on
# layered lattices (mixed.py); last, the best loop is polished off any grid
# (polish.py)
CIRCUIT_BUDGET = 60.0
REFINE_BUDGET = 150.0  # rebuilding the loop's edges on fine grids outside it
ADAPTIVE_BUDGET = 185.0
MIXED_BUDGET = 235.0
LAST_REFINE_BUDGET = 245.0
POLISH_SEEDS = 4  # loops polished: the best and runners-up of other kinds
POLISH_BUDGET = 255.0  # the 300 s limit also covers the simulator's own work
CIRCUIT_SOLVE_SECONDS = 30.0  # the most one lattice may take
FEASIBILITY_SECONDS = 2.0
# lattice spacings, tried in this order; a lattice is skipped when it has more
# nodes or sides than CP-SAT reliably handles in a few seconds. The coarse ones
# are what big rooms need: a 400 x 400 room is far over the caps at spacing 5
# but solves to a proven optimum in seconds at spacing 10
SPACINGS = (10, 5, 15, 3, 2, 1, 4)
MAX_LATTICE_NODES = 1200
MAX_LATTICE_SIDES = 10000
LATTICE_HEADINGS = 2  # the room's main wall directions to align lattices with
LATTICE_ORIGINS = 2  # room corners to anchor lattices at
# layered lattices for 45-degree corners: one diagonal side length per lattice,
# layers up to MIXED_MOVES diagonal sides from the plain lattice
MIXED_SPACINGS = (5, 3, 2, 1, 10)
MIXED_DIAGONAL_LENGTHS = 6  # the shortest buildable lengths tried as diagonals
MIXED_MOVES = 2
MAX_MIXED_NODES = 1500
MAX_MIXED_SIDES = 12000
MIXED_SOLVE_SECONDS = 15.0
# adaptive lattices: rooms with more nodes than this are covered window by window
MAX_ADAPTIVE_NODES = 900
ADAPTIVE_WINDOW = 50.0
ADAPTIVE_SOLVE_SECONDS = 60.0

Candidate = tuple[float, str, tuple[int, ...]]


class Player10(Player):
    """Group 10: first the best-scoring rectangle, octagon, L-shape or U-shape the
    inventory can build (sides may be several pieces), placed against the room's own
    walls; then any right-angled loop that beats it, found by CP-SAT on lattices laid
    over the room (circuit.py)."""

    def build_enclosure(self) -> Construction | None:
        self._best: Construction | None = None
        self._best_score = 0.0  # a valid enclosure can score below 0; None is 0
        self._seeds: dict[str, tuple[float, Construction]] = {}
        try:
            self._build()
        except Exception as e:  # noqa: BLE001 -- a crash would end the whole run
            self.note = f"search stopped by {type(e).__name__}: {e}"
        return self._best

    def _build(self) -> None:
        started = time.process_time()
        weights = self.weights
        self._assigner = Assigner(self.inventory, want_second_gate=weights.G > 0)
        if self._assigner.gates == 0:
            self.note = "no usable gate in the inventory -- not attempting a build"
            return
        self._search_room = RoomSearch(self.room)
        self._failures: dict[str, list[tuple[int, ...]]] = {}
        self._fitting_rectangles: list[tuple[int, int]] = []
        bonus = weights.G if len(self._assigner.targets) == 2 else 0.0
        families = [
            family
            for family, (kind, needed) in CORNERS_NEEDED.items()
            if self.inventory.connectors.get(kind, 0) >= needed
        ]

        # cheapest families first, so there is always something to fall back on
        # and a score for the dearer searches to beat. Octagons are slow to rule
        # out; U-shapes are many, and only pay off where nothing else fits.
        stages = [
            ([RECTANGLE], RECTANGLE_SHARE),
            ([L_SHAPE], L_SHARE),
            ([OCTAGON], OCTAGON_SHARE),
            ([U_SHAPE], 1.0),
        ]
        for stage, share in stages:
            wanted = [f for f in stage if f in families]
            if wanted:
                self._deadline = started + TIME_BUDGET * share
                self._search([x for f in wanted for x in self._candidates(f, bonus)])
        self._lattice_stage(started + CIRCUIT_BUDGET)
        self._held_templates(families, bonus)
        # fine lattices first only where the best loop leaves room uncovered; time
        # they leave goes to fine lattices over the whole room, and whatever loop
        # those find gets the same treatment at the end
        # A stage runs only while it could still beat the best loop: nothing can
        # once the score reaches its upper bound, and no right-angled loop can
        # once the unit lattice is solved to optimality in a room whose walls
        # all lie on it (see _lattice_stage).
        self._refiner = None
        if not self._settled(right_angled=True):
            self._refine_stage("adaptive", started + REFINE_BUDGET)
        if not self._settled(right_angled=True):
            self._adaptive_stage(started + ADAPTIVE_BUDGET)
        if not self._settled():
            self._mixed_stage(started + MIXED_BUDGET)
        self._merge_held()
        if not self._settled() and (
            not self._settled(right_angled=True) or self._best_has_diagonals()
        ):
            self._refine_stage("adaptive", started + LAST_REFINE_BUDGET)
        if not self._settled():
            self._polish(started + POLISH_BUDGET)
        if self._best is None and not self.note:
            self.note = "nothing the inventory can build fits and scores above 0"

    def _held_templates(self, families, bonus: float) -> None:
        """Hexagons, then octagons again with the hexagon's score to beat (fewer
        to try, so it reaches ones the first pass ran out of time for). The
        winner is held back until the lattice stages are done: a strong shape
        here raises the score every lattice search must beat, and CP-SAT then
        settles on loops that polish worse."""
        self._held = None
        if HEXAGON not in families:
            return
        saved = (self._best, self._best_score, self.note)
        for family, seconds in ((HEXAGON, HEXAGON_SECONDS), (OCTAGON, HELD_OCTAGON_SECONDS)):
            if family not in families:
                continue
            if family == OCTAGON and self._best_score <= saved[1]:
                break  # no hexagon win: the first octagon pass already covered this
            self._deadline = time.process_time() + seconds
            self._search(self._candidates(family, bonus))
        if self._best_score > saved[1]:
            self._held = (self._best, self._best_score, self.note)
        self._best, self._best_score, self.note = saved

    def _merge_held(self) -> None:
        held = getattr(self, "_held", None)
        if held is not None and held[1] > self._best_score:
            self._best, self._best_score, self.note = held

    # --------------------------------------------------------------- lattices

    def _lattice_stage(self, deadline: float) -> None:
        """Right-angled loops on lattices over the room, each solved by CP-SAT and
        required to beat the best enclosure so far."""
        rights = self.inventory.connectors.get(ConnectorType.RIGHT, 0)
        if not circuit.AVAILABLE or rights < 4:
            return
        if not circuit.loop_possible(self._assigner, rights, FEASIBILITY_SECONDS):
            return
        polygon = self.room.polygon
        for spacing, heading, origin in self._lattices():
            remaining = deadline - time.process_time()
            if remaining < 1.0:
                return
            if (
                circuit.count_nodes(polygon, heading, spacing, origin)
                > MAX_LATTICE_NODES
            ):
                continue
            lattice = circuit.build_lattice(
                polygon, heading, spacing, origin, self._assigner.lengths
            )
            if len(lattice.sides) > MAX_LATTICE_SIDES:
                continue
            seconds = min(CIRCUIT_SOLVE_SECONDS, deadline - time.process_time())
            found, status = circuit.solve(
                lattice, self._assigner, self.weights, rights, self._best_score, seconds
            )
            if found is not None:
                self._keep(found, f"right-angled loop on a spacing-{spacing} lattice")
            if (
                spacing == 1
                and status in ("OPTIMAL", "INFEASIBLE")
                and self._walls_on_lattice(heading, origin)
            ):
                # every right-angled loop with whole-length sides in such a room
                # fits on this lattice (shift it to whole coordinates: each unit
                # cell it covered is still in the room), so none beats this one
                self._right_angled_proven = True
                return

    def _adaptive_stage(self, deadline: float) -> None:
        """Right-angled loops on adaptive lattices (adaptive.py): the whole room
        if it is small enough, else window by window."""
        rights = self.inventory.connectors.get(ConnectorType.RIGHT, 0)
        if not circuit.AVAILABLE or rights < 4:
            return
        polygon = self.room.polygon
        corners = self._search_room.corners
        origin = tuple(float(v) for v in corners[self._search_room._long_edges[0]])
        lengths = self._assigner.lengths
        for heading in self._search_room.base_headings(90)[:LATTICE_HEADINGS]:
            whole = adaptive.build_adaptive_lattice(
                polygon, heading, origin, lengths, MAX_ADAPTIVE_NODES
            )
            if whole is not None:
                lattices = [whole]
            else:
                lattices = (
                    adaptive.build_adaptive_lattice(
                        polygon, heading, origin, lengths, MAX_ADAPTIVE_NODES, box
                    )
                    for box in adaptive.windows(
                        polygon, heading, origin, ADAPTIVE_WINDOW
                    )
                )
            for lattice in lattices:
                if deadline - time.process_time() < 1.0:
                    return
                if lattice is not None:
                    self._solve_adaptive(lattice, rights, deadline)

    def _solve_adaptive(self, lattice, rights: int, deadline: float) -> None:
        """Cell area first: its tight bound often proves the best loop quickly.
        If it does not, the shoelace form gets what time is left, as it finds
        good loops in some rooms where the cell form stalls."""
        diagonals = self.inventory.connectors.get(ConnectorType.DIAGONAL, 0)
        for use_cells in (True, False):
            seconds = min(ADAPTIVE_SOLVE_SECONDS, deadline - time.process_time())
            if seconds < 1.0:
                return
            if not use_cells:
                lattice.cells = None
            found, status = mixed.solve_mixed(
                lattice,
                self._assigner,
                self.weights,
                rights,
                diagonals,
                self._best_score,
                seconds,
            )
            if found is not None:
                self._keep(found, "right-angled loop on an adaptive lattice")
            if status in ("OPTIMAL", "INFEASIBLE"):
                return  # proven: nothing better on this lattice

    def _mixed_stage(self, deadline: float) -> None:
        """Loops mixing 45- and 90-degree corners on layered lattices (mixed.py),
        each required to beat the best enclosure so far."""
        rights = self.inventory.connectors.get(ConnectorType.RIGHT, 0)
        diagonals = self.inventory.connectors.get(ConnectorType.DIAGONAL, 0)
        if not circuit.AVAILABLE or diagonals < 2:
            return
        polygon = self.room.polygon
        headings = self._search_room.base_headings(45)[:LATTICE_HEADINGS]
        corners = self._search_room.corners
        origin = tuple(float(v) for v in corners[self._search_room._long_edges[0]])
        lengths = self._assigner.lengths
        for spacing in MIXED_SPACINGS:
            for heading in headings:
                for diagonal in lengths[:MIXED_DIAGONAL_LENGTHS]:
                    if deadline - time.process_time() < 1.0:
                        return
                    lattice = mixed.build_mixed_lattice(
                        polygon,
                        heading,
                        spacing,
                        origin,
                        lengths,
                        [diagonal],
                        MIXED_MOVES,
                        MAX_MIXED_NODES,
                    )
                    if lattice is None or len(lattice.sides) > MAX_MIXED_SIDES:
                        continue
                    seconds = min(MIXED_SOLVE_SECONDS, deadline - time.process_time())
                    found, _ = mixed.solve_mixed(
                        lattice,
                        self._assigner,
                        self.weights,
                        rights,
                        diagonals,
                        self._best_score,
                        seconds,
                    )
                    if found is not None:
                        self._keep(
                            found,
                            f"loop with 45-degree corners on a spacing-{spacing} "
                            f"layered lattice (diagonals {diagonal})",
                        )

    def _walls_on_lattice(self, heading: float, origin) -> bool:
        """Whether every wall runs along the unit lattice with this heading and
        origin: the room is a union of its unit cells."""
        local = circuit.local_room(self.room.polygon, heading, origin)
        for x, y in local.exterior.coords:
            if abs(x - round(x)) > 1e-6 or abs(y - round(y)) > 1e-6:
                return False
        points = list(local.exterior.coords)
        return all(
            abs(ax - bx) < 1e-6 or abs(ay - by) < 1e-6
            for (ax, ay), (bx, by) in zip(points, points[1:])
        )

    def _upper_bound(self) -> float:
        """No loop scores more: the room's whole area, all the inventory's length
        as perimeter (if perimeter pays), the gate bonus."""
        w = self.weights
        length = sum(key[1] * n for key, n in self._assigner.stock.items())
        return (
            c.BASELINE_SCORE
            + max(0.0, w.A) * self.room.polygon.area
            + max(0.0, w.C) * length
            + max(0.0, w.G)
        )

    def _settled(self, right_angled: bool = False) -> bool:
        """Whether no later stage can beat the best loop: no stage at all, or
        (with `right_angled`) none limited to right-angled loops."""
        if self._best is not None and self._best_score >= self._upper_bound() - 1e-6:
            return True
        return right_angled and getattr(self, "_right_angled_proven", False)

    def _best_has_diagonals(self) -> bool:
        return self._best is not None and any(
            k.connector_type == ConnectorType.DIAGONAL for k in self._best.connectors
        )

    def _refine_stage(self, kind: str, deadline: float) -> None:
        """Rebuild stretches of the best loop on `kind` lattices laid over the
        gaps it leaves (refine.py)."""
        if self._best is None or not circuit.AVAILABLE:
            return
        if kind == "layered" and self.inventory.connectors.get(ConnectorType.DIAGONAL, 0) < 1:
            return
        if self._refiner is None:  # one refiner, so windows that failed stay skipped
            self._refiner = refine.Refiner(
                self.room, self.inventory, self.weights, self._assigner
            )
        best, score = self._refiner.run(self._best, kind, deadline)
        if score is not None and score > self._best_score + 1e-6:
            self._best, self._best_score = best, score
            grid = "an adaptive" if kind == "adaptive" else "a 45-degree"
            self.note = f"{self.note}; rebuilt on {grid} grid outside it to {score:.1f}"

    def _polish(self, deadline: float) -> None:
        """Slide parts of the loop off any grid, put opposite diagonal pairs on
        it, and shift it so corners meet the walls, while any of that pays
        (polish.py). The best loop goes first, then a few runners-up of other
        kinds: a rectangle with two cut corners can beat the L that beat it."""
        if self._best is None:
            return
        seeds = sorted(self._seeds.values(), key=lambda e: -e[0])
        starts = [self._best] + [
            c_ for _, c_ in seeds if c_ is not self._best
        ][: POLISH_SEEDS - 1]
        for n, start in enumerate(starts):
            left = deadline - time.process_time()
            if left < 0.5:
                return
            # the best loop gets half the time, the others share the rest
            share = left / 2 if n == 0 else left / (len(starts) - n)
            best, score = self._polish_one(start, time.process_time() + share)
            if score is not None and score > self._best_score + 1e-6:
                self._best, self._best_score = best, score
                self.note = f"{self.note}; polished to {score:.1f}"

    def _polish_one(self, construction, deadline: float):
        args = (self.room, self.inventory, self.weights, self._assigner, deadline)
        result = validate_construction(construction, self.room, self.inventory)
        if not result.valid:
            return construction, None
        best, best_score = construction, score_construction(result, self.weights)
        moves = (polish.polish, polish.chamfer_pairs, polish.shift_to_walls)
        improved = True
        while improved and time.process_time() < deadline:
            improved = False
            for move in moves:
                found, score = move(best, *args)
                if score is not None and score > best_score + 1e-6:
                    best, best_score, improved = found, score, True
        return best, best_score

    def _seed(self, construction, score: float, what: str) -> None:
        """Remember the best loop of each kind: one that scores less now may
        win once polishing puts diagonals on it."""
        kind = what.split(",")[0].split(" ")[0]  # rectangle, L, octagon, right-angled...
        if kind not in self._seeds or score > self._seeds[kind][0]:
            self._seeds[kind] = (score, construction)

    def _keep(self, found, what: str) -> None:
        """Validate a lattice result with the simulator and keep it if it is the
        best so far."""
        result = validate_construction(found.construction, self.room, self.inventory)
        if not result.valid:
            return
        score = score_construction(result, self.weights)
        self._seed(found.construction, score, what)
        if score > self._best_score:
            self._best, self._best_score = found.construction, score
            self.note = (
                f"{what}, {len(found.construction.pieces)} pieces, score {score:.1f}"
                + (" (best on that lattice)" if found.proven else "")
            )

    def _lattices(self) -> list[tuple[int, float, tuple[float, float]]]:
        """(spacing, heading, origin) for every lattice worth a try, coarse first,
        each distinct lattice once."""
        headings = self._search_room.base_headings(90)[:LATTICE_HEADINGS]
        corners = self._search_room.corners
        edges = self._search_room._long_edges[:LATTICE_ORIGINS]
        origins = [tuple(float(v) for v in corners[e]) for e in edges]
        seen, out = set(), []
        for spacing in SPACINGS:
            for heading in headings:
                t = math.radians(heading)
                for ox, oy in origins:
                    # nodes of two lattices coincide when their origins differ by
                    # whole steps along the lattice axes
                    u = (ox * math.cos(t) + oy * math.sin(t)) % spacing
                    v = (-ox * math.sin(t) + oy * math.cos(t)) % spacing
                    key = (
                        spacing,
                        round(heading, 6),
                        round(u, 6) % spacing,
                        round(v, 6) % spacing,
                    )
                    if key not in seen:
                        seen.add(key)
                        out.append((spacing, heading, (ox, oy)))
        return out

    # ------------------------------------------------------------ candidates

    def _candidates(self, family: str, bonus: float) -> list[Candidate]:
        """Every shape of a family worth trying, with the most it could score."""
        lengths = self._assigner.lengths
        usable = set(lengths)
        if family == RECTANGLE:
            params = [(a, b) for a in lengths for b in lengths if b <= a]
        elif family == OCTAGON:
            # p is the longest side: the other choices are the same octagon turned
            params = [
                (p, r, q, t)
                for p in lengths
                for r in lengths
                if r <= p
                for q in lengths
                if q <= p
                for t in lengths
                if t <= p
            ]
        elif family == HEXAGON:
            params = [(p, r, q) for p in lengths for r in lengths for q in lengths]
        elif family == U_SHAPE:
            params = [
                (left, gap, right, height, leg)
                for left in lengths
                for gap in lengths
                for right in lengths
                if left + gap + right in usable
                for height in lengths
                for leg in lengths
                if height - leg >= c.MIN_WALL_LENGTH
            ]
        else:
            params = [
                (a, b, cut, d)
                for a in lengths
                for cut in lengths
                if a - cut in usable
                for b in lengths
                for d in lengths
                if b + d in usable
            ]
        measure = MEASURES[family]
        out = []
        for p in params:
            area, perimeter = measure(*p)
            if area > self._search_room.area + c.TOL:
                continue
            score = self._score(area, perimeter) + bonus
            if score > self._best_score:
                out.append((score, family, p))
        return out

    def _score(self, area: float, perimeter: float) -> float:
        return c.BASELINE_SCORE + self.weights.A * area + self.weights.C * perimeter

    # ---------------------------------------------------------------- search

    def _search(self, candidates: list[Candidate]) -> None:
        """Try candidates from the highest score down; the first that can be built
        and placed is the best of them."""
        # (-score, tie-break, family, params, score is exact)
        heap = [(-s, i, f, p, False) for i, (s, f, p) in enumerate(candidates)]
        heapq.heapify(heap)
        counter = len(heap)
        timed_out = False

        while heap:
            if time.process_time() > self._deadline:
                timed_out = True
                break
            negative, _, family, params, exact = heapq.heappop(heap)
            if -negative <= self._best_score:
                return
            if self._cannot_fit(family, params):
                continue
            shape = make_shape(family, params)
            assignment = self._assigner.assign(shape.faces)
            if assignment is None:
                continue
            if not exact:
                score = self._score(shape.area, shape.perimeter)
                if assignment.gate_faces >= 2:
                    score += self.weights.G
                if score < -negative - 1e-9:
                    # no second gate after all: it is worth less than it was
                    # ranked at, so let it wait its turn
                    counter += 1
                    heapq.heappush(heap, (-score, counter, family, params, True))
                    continue
            if self._try(shape, assignment):
                return
            self._note_failure(family, params)

        if timed_out and self._best is None:
            # ran out of time on big shapes: settle for the smallest ones
            deadline = time.process_time() + FALLBACK_BUDGET
            for _, _, family, params, _ in heapq.nlargest(FALLBACK_SHAPES, heap):
                if time.process_time() > deadline:
                    break
                shape = make_shape(family, params)
                assignment = self._assigner.assign(shape.faces)
                if assignment is not None:
                    self._try(shape, assignment)

    def _cannot_fit(self, family: str, params: tuple[int, ...]) -> bool:
        """Cheap reasons a shape can't be placed, tried before the real search."""
        if inner_radius(family, params) > self._search_room.inradius:
            return True
        recent = self._failures.get(family, [])[-RECENT_FAILURES:]
        if any(contains_smaller(family, params, small) for small in recent):
            return True
        return not all(
            self._rectangle_fits(max(r), min(r))
            for r in inner_rectangles(family, params)
        )

    def _rectangle_fits(self, a: int, b: int) -> bool:
        """Whether an a x b rectangle (a >= b) fits anywhere in the room, whether
        or not the inventory can build it. Remembered, and inferred from other
        rectangles where possible."""
        if any(fa >= a and fb >= b for fa, fb in self._fitting_rectangles):
            return True
        failures = self._failures.setdefault(RECTANGLE, [])
        if any(a >= fa and b >= fb for fa, fb in failures):
            return False
        if inner_radius(RECTANGLE, (a, b)) > self._search_room.inradius:
            return False
        fits = self._placements(make_shape(RECTANGLE, (a, b)), limit=1) != []
        (self._fitting_rectangles if fits else failures).append((a, b))
        return fits

    def _try(self, shape: Shape, assignment: Assignment) -> bool:
        """Place the shape; keep it if it is valid and beats the best so far."""
        for start, heading in self._placements(shape):
            construction = self._construction(shape, assignment, start, heading)
            # the simulator's own judge: only what it accepts is ever returned
            result = validate_construction(construction, self.room, self.inventory)
            if not result.valid:
                continue
            score = score_construction(result, self.weights)
            self._seed(construction, score, shape.describe())
            if score > self._best_score:
                self._best, self._best_score = construction, score
                self.note = f"{shape.describe()}, score {score:.1f}"
            return True
        return False

    def _placements(self, shape: Shape, limit: int = 3) -> list:
        return self._search_room.place(
            shape.vertices(),
            HEADING_STEP_DEG[shape.family],
            SYMMETRY_DEG[shape.family],
            limit,
        )

    def _note_failure(self, family: str, params: tuple[int, ...]) -> None:
        """Remember that a shape fits nowhere. Shapes are tried biggest first, so a
        failure on its own rules out little; a much smaller version that also
        fails rules out everything that could hold it."""
        failures = self._failures.setdefault(family, [])
        smallest = params
        if family in (RECTANGLE, OCTAGON, HEXAGON):
            fitting, failing = 0.0, 1.0
            for _ in range(SHRINK_STEPS):
                scale = (fitting + failing) / 2.0
                shrunk = tuple(max(c.MIN_WALL_LENGTH, round(scale * p)) for p in params)
                if shrunk == smallest:
                    break
                if family == RECTANGLE:
                    fits = self._rectangle_fits(max(shrunk), min(shrunk))
                else:
                    fits = self._placements(make_shape(family, shrunk), 1) != []
                if fits:
                    fitting = scale
                else:
                    failing, smallest = scale, shrunk
        if smallest not in failures:
            failures.append(smallest)

    @staticmethod
    def _construction(
        shape: Shape,
        assignment: Assignment,
        start: tuple[float, float],
        heading: float,
    ) -> Construction:
        pieces, connectors = [], []
        for face, keys in enumerate(assignment.faces):
            for i, (is_gate, length) in enumerate(keys):
                kind = PieceType.GATE if is_gate else PieceType.WALL
                pieces.append(Piece(kind, length))
                connectors.append(
                    shape.corner(face) if i == 0 else Connector(ConnectorType.STRAIGHT)
                )
        return Construction(start, heading, pieces, connectors)
