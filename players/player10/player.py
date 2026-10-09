import math
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass

from shapely.affinity import rotate, translate
from shapely.geometry import Polygon
from shapely.prepared import prep

import src.constants as c
from players.player0 import Player
from src.enclosure import (
    Construction,
    ValidationResult,
    score_construction,
    validate_construction,
)
from src.pieces import Connector, ConnectorType, Piece, PieceType

HEADINGS_DEG = tuple(range(0, 180, 15))
FINE_HEADINGS_DEG = tuple(range(0, 180, 5))
GRID_STEPS = 9
MAX_TEMPLATE_GEOMETRIES = 40
MAX_TEMPLATE_GEOMETRY_ATTEMPTS = 400
MAX_GENERAL_FACE_OPTIONS_PER_GATE_CLASS = 4


@dataclass(frozen=True)
class FaceOption:
    pieces: tuple[Piece, ...]
    usage: tuple[int, ...]
    length: int
    has_gate: bool


@dataclass(frozen=True)
class ShapeCandidate:
    width: float
    height: float
    faces: tuple[tuple[Piece, ...], ...]
    score_hint: float
    kind: str = "rectangle"
    local_vertices: tuple[tuple[float, float], ...] = ()
    face_connectors: tuple[Connector, ...] = ()

    @property
    def gate_count(self) -> int:
        return sum(piece.is_gate for face in self.faces for piece in face)

    @property
    def gate_face_count(self) -> int:
        return sum(any(piece.is_gate for piece in face) for face in self.faces)

    @property
    def piece_count(self) -> int:
        return sum(map(len, self.faces))

    @property
    def polygon(self) -> Polygon:
        if self.local_vertices:
            return Polygon(self.local_vertices)
        return Polygon(
            (
                (0.0, 0.0),
                (self.width, 0.0),
                (self.width, self.height),
                (0.0, self.height),
            )
        )

    @property
    def area(self) -> float:
        return self.polygon.area

    @staticmethod
    def _dimension(value: float) -> str:
        return (
            str(round(value)) if math.isclose(value, round(value)) else f"{value:.1f}"
        )

    @property
    def name(self) -> str:
        gate_word = "gate" if self.gate_count == 1 else "gates"
        dimensions = f"{self._dimension(self.width)}x{self._dimension(self.height)}"
        prefix = "" if self.kind == "rectangle" else f"{self.kind} "
        return (
            f"{prefix}{dimensions} "
            f"({self.gate_count} {gate_word}, {self.piece_count} pieces)"
        )


class Player10(Player):
    """Score-first rectangle and bounded multi-face template search."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._candidate_cache: tuple[ShapeCandidate, ...] | None = None
        self._headings_cache: tuple[float, ...] | None = None
        self._edge_headings_cache: tuple[float, ...] | None = None

    def _piece_kinds(self) -> tuple[tuple[PieceType, int, int], ...]:
        kinds = []
        for piece_type, pool in (
            (PieceType.GATE, self.inventory.gates),
            (PieceType.WALL, self.inventory.walls),
        ):
            for length, count in sorted(pool.items()):
                if count > 0 and c.MIN_WALL_LENGTH <= length <= c.MAX_FACE_LENGTH:
                    kinds.append((piece_type, length, count))
        return tuple(kinds)

    def _face_options(
        self, kinds: tuple[tuple[PieceType, int, int], ...]
    ) -> dict[int, list[FaceOption]]:
        """All inventory-bounded multisets that can form one linear face."""
        by_length: dict[int, list[FaceOption]] = {}
        usage = [0] * len(kinds)

        def search(start: int, total: int) -> None:
            if total:
                pieces = tuple(
                    Piece(piece_type, length)
                    for i, (piece_type, length, _) in enumerate(kinds)
                    for _ in range(usage[i])
                )
                option = FaceOption(
                    pieces=pieces,
                    usage=tuple(usage),
                    length=total,
                    has_gate=any(piece.is_gate for piece in pieces),
                )
                by_length.setdefault(total, []).append(option)

            for i in range(start, len(kinds)):
                _, length, available = kinds[i]
                max_take = min(available, (c.MAX_FACE_LENGTH - total) // length)
                for take in range(1, max_take + 1):
                    usage[i] = take
                    search(i + 1, total + take * length)
                usage[i] = 0

        search(0, 0)
        for options in by_length.values():
            options.sort(
                key=lambda option: (
                    len(option.pieces),
                    not option.has_gate,
                    tuple(
                        (piece.piece_type.value, piece.length)
                        for piece in option.pieces
                    ),
                )
            )
        return by_length

    def _faces_for_rectangle(
        self,
        width: int,
        height: int,
        want_bonus: bool,
        kinds: tuple[tuple[PieceType, int, int], ...],
        options_by_length: dict[int, list[FaceOption]],
    ) -> tuple[tuple[Piece, ...], ...] | None:
        return self._faces_for_lengths(
            (width, height, width, height), want_bonus, kinds, options_by_length
        )

    def _faces_for_lengths(
        self,
        lengths: tuple[int, ...],
        want_bonus: bool,
        kinds: tuple[tuple[PieceType, int, int], ...],
        options_by_length: dict[int, list[FaceOption]],
    ) -> tuple[tuple[Piece, ...], ...] | None:
        """Assign inventory-compatible pieces to an arbitrary face sequence."""
        if any(length not in options_by_length for length in lengths):
            return None

        available = tuple(kind[2] for kind in kinds)
        straight_available = self.inventory.connectors.get(ConnectorType.STRAIGHT, 0)
        failed: set[tuple[int, tuple[int, ...], int]] = set()

        def search(
            face_index: int,
            remaining: tuple[int, ...],
            gate_faces: int,
            selected: tuple[tuple[Piece, ...], ...],
        ) -> tuple[tuple[Piece, ...], ...] | None:
            state = (face_index, remaining, min(gate_faces, 2))
            if state in failed:
                return None
            if face_index == len(lengths):
                gate_requirement_met = (
                    gate_faces >= 2 if want_bonus else gate_faces == 1
                )
                piece_count = sum(map(len, selected))
                if (
                    gate_requirement_met
                    and piece_count - len(lengths) <= straight_available
                ):
                    return selected
                failed.add(state)
                return None

            faces_left = len(lengths) - face_index
            if want_bonus and gate_faces + faces_left < 2:
                failed.add(state)
                return None

            need_gate = want_bonus and gate_faces < 2
            ordered = sorted(
                options_by_length[lengths[face_index]],
                key=lambda option: (
                    option.has_gate != need_gate,
                    len(option.pieces),
                ),
            )
            for option in ordered:
                next_gate_faces = gate_faces + option.has_gate
                if not want_bonus and next_gate_faces > 1:
                    continue
                if any(used > left for used, left in zip(option.usage, remaining)):
                    continue
                next_remaining = tuple(
                    left - used for used, left in zip(option.usage, remaining)
                )
                pieces_used = sum(available) - sum(next_remaining)
                if pieces_used - (face_index + 1) > straight_available:
                    continue
                result = search(
                    face_index + 1,
                    next_remaining,
                    next_gate_faces,
                    (*selected, option.pieces),
                )
                if result is not None:
                    return result

            failed.add(state)
            return None

        return search(0, available, 0, ())

    @staticmethod
    def _right_connectors(count: int, reflex_indices: set[int] | None = None):
        reflex_indices = reflex_indices or set()
        return tuple(
            Connector(ConnectorType.RIGHT, reflex=index in reflex_indices)
            for index in range(count)
        )

    def _candidate_for_geometry(
        self,
        kind: str,
        vertices: tuple[tuple[float, float], ...],
        face_lengths: tuple[int, ...],
        face_connectors: tuple[Connector, ...],
        kinds: tuple[tuple[PieceType, int, int], ...],
        options_by_length: dict[int, list[FaceOption]],
    ) -> ShapeCandidate | None:
        connector_needs = Counter(
            connector.connector_type for connector in face_connectors
        )
        if any(
            self.inventory.connectors.get(connector_type, 0) < needed
            for connector_type, needed in connector_needs.items()
        ):
            return None

        polygon = Polygon(vertices)
        if (
            not polygon.is_valid
            or polygon.area <= 0
            or polygon.area > self.room.polygon.area + c.TOL
        ):
            return None
        minx, miny, maxx, maxy = polygon.bounds
        choices = []
        for want_bonus in (False, True):
            faces = self._faces_for_lengths(
                face_lengths, want_bonus, kinds, options_by_length
            )
            if faces is None:
                continue
            score = (
                c.BASELINE_SCORE
                + self.weights.A * polygon.area
                + self.weights.C * polygon.length
                + (self.weights.G if want_bonus else 0.0)
            )
            choices.append((score, -sum(map(len, faces)), faces))
        if not choices:
            return None
        score, _, faces = max(choices, key=lambda choice: (choice[0], choice[1]))
        return ShapeCandidate(
            width=maxx - minx,
            height=maxy - miny,
            faces=faces,
            score_hint=score,
            kind=kind,
            local_vertices=vertices,
            face_connectors=face_connectors,
        )

    def _general_candidates(
        self,
        kinds: tuple[tuple[PieceType, int, int], ...],
        options_by_length: dict[int, list[FaceOption]],
    ) -> list[ShapeCandidate]:
        """Generate score-first L, U, and diagonal-octagon templates."""
        lengths = tuple(sorted(options_by_length))
        length_set = set(lengths)
        geometries = []
        # General polygons have more faces than rectangles. Keep the shortest
        # gate and wall compositions for each total so inventory assignment
        # remains bounded while preserving both gate choices.
        general_options = {
            length: [
                *[option for option in options if option.has_gate][
                    :MAX_GENERAL_FACE_OPTIONS_PER_GATE_CLASS
                ],
                *[option for option in options if not option.has_gate][
                    :MAX_GENERAL_FACE_OPTIONS_PER_GATE_CLASS
                ],
            ]
            for length, options in options_by_length.items()
        }

        def projected(area: float, perimeter: float) -> float:
            return (
                c.BASELINE_SCORE
                + self.weights.A * area
                + self.weights.C * perimeter
                + self.weights.G
            )

        right_available = self.inventory.connectors.get(ConnectorType.RIGHT, 0)
        if right_available >= 6:
            splits = [
                (total, first, total - first)
                for total in lengths
                for first in lengths
                if total - first in length_set
            ]
            l_params = []
            for width, left, right in splits:
                for height, bottom, top in splits:
                    area = width * bottom + left * top
                    perimeter = 2 * (width + height)
                    l_params.append(
                        (projected(area, perimeter), width, height, left, bottom)
                    )
            l_params.sort(reverse=True)
            for _, width, height, left, bottom in l_params[
                :MAX_TEMPLATE_GEOMETRY_ATTEMPTS
            ]:
                right = width - left
                top = height - bottom
                vertices = (
                    (0.0, 0.0),
                    (float(width), 0.0),
                    (float(width), float(bottom)),
                    (float(left), float(bottom)),
                    (float(left), float(height)),
                    (0.0, float(height)),
                )
                geometries.append(
                    (
                        "L",
                        vertices,
                        (width, bottom, right, top, left, height),
                        self._right_connectors(6, {3}),
                    )
                )

        if right_available >= 8:
            u_params = []
            for arm in lengths:
                for gap in lengths:
                    width = 2 * arm + gap
                    if width not in length_set:
                        continue
                    for height in lengths:
                        for leg in lengths:
                            if height - leg < c.MIN_WALL_LENGTH:
                                continue
                            area = width * height - gap * leg
                            perimeter = 2 * (width + height + leg)
                            u_params.append(
                                (
                                    projected(area, perimeter),
                                    width,
                                    height,
                                    arm,
                                    gap,
                                    leg,
                                )
                            )
            u_params.sort(reverse=True)
            for _, width, height, arm, gap, leg in u_params[
                :MAX_TEMPLATE_GEOMETRY_ATTEMPTS
            ]:
                notch_bottom = height - leg
                vertices = (
                    (0.0, 0.0),
                    (float(width), 0.0),
                    (float(width), float(height)),
                    (float(width - arm), float(height)),
                    (float(width - arm), float(notch_bottom)),
                    (float(arm), float(notch_bottom)),
                    (float(arm), float(height)),
                    (0.0, float(height)),
                )
                geometries.append(
                    (
                        "U",
                        vertices,
                        (width, height, arm, leg, gap, leg, arm, height),
                        self._right_connectors(8, {4, 5}),
                    )
                )

        if self.inventory.connectors.get(ConnectorType.DIAGONAL, 0) >= 8:
            octagon_params = []
            for horizontal in lengths:
                for vertical in lengths:
                    for diagonal in lengths:
                        face_lengths = (
                            horizontal,
                            diagonal,
                            vertical,
                            diagonal,
                            horizontal,
                            diagonal,
                            vertical,
                            diagonal,
                        )
                        x = y = 0.0
                        vertices = []
                        for angle, length in zip(range(0, 360, 45), face_lengths):
                            vertices.append((x, y))
                            theta = math.radians(angle)
                            x += length * math.cos(theta)
                            y += length * math.sin(theta)
                        polygon = Polygon(vertices)
                        octagon_params.append(
                            (
                                projected(polygon.area, polygon.length),
                                tuple(vertices),
                                face_lengths,
                            )
                        )
            octagon_params.sort(key=lambda item: item[0], reverse=True)
            diagonal_connectors = tuple(
                Connector(ConnectorType.DIAGONAL) for _ in range(8)
            )
            for _, vertices, face_lengths in octagon_params[
                :MAX_TEMPLATE_GEOMETRY_ATTEMPTS
            ]:
                geometries.append(
                    ("octagon", vertices, face_lengths, diagonal_connectors)
                )

        candidates = []
        candidate_counts = Counter()
        for kind, vertices, face_lengths, connectors in geometries:
            if candidate_counts[kind] >= MAX_TEMPLATE_GEOMETRIES:
                continue
            candidate = self._candidate_for_geometry(
                kind,
                vertices,
                face_lengths,
                connectors,
                kinds,
                general_options,
            )
            if candidate is not None:
                candidates.append(candidate)
                candidate_counts[kind] += 1
        return candidates

    def _candidate_shapes(self) -> list[ShapeCandidate]:
        if self._candidate_cache is not None:
            return list(self._candidate_cache)
        right_available = self.inventory.connectors.get(ConnectorType.RIGHT, 0)
        diagonal_available = self.inventory.connectors.get(ConnectorType.DIAGONAL, 0)
        if right_available < 4 and diagonal_available < 8:
            return []

        kinds = self._piece_kinds()
        options_by_length = self._face_options(kinds)
        candidates = []
        lengths = sorted(options_by_length)
        if right_available >= 4:
            for width in lengths:
                for height in lengths:
                    if height > width:
                        continue
                    plain_faces = self._faces_for_rectangle(
                        width, height, False, kinds, options_by_length
                    )
                    bonus_faces = self._faces_for_rectangle(
                        width, height, True, kinds, options_by_length
                    )
                    choices = []
                    if plain_faces is not None:
                        choices.append((plain_faces, False))
                    if bonus_faces is not None:
                        choices.append((bonus_faces, True))
                    if not choices:
                        continue

                    def choice_key(
                        choice: tuple[tuple[tuple[Piece, ...], ...], bool],
                    ):
                        faces, bonus = choice
                        return (
                            self.weights.G if bonus else 0.0,
                            -sum(map(len, faces)),
                        )

                    faces, bonus = max(choices, key=choice_key)
                    score_hint = (
                        c.BASELINE_SCORE
                        + self.weights.A * width * height
                        + self.weights.C * 2 * (width + height)
                        + (self.weights.G if bonus else 0.0)
                    )
                    candidates.append(ShapeCandidate(width, height, faces, score_hint))

        candidates.extend(self._general_candidates(kinds, options_by_length))

        candidates.sort(
            key=lambda candidate: (
                -candidate.score_hint,
                -candidate.area,
                candidate.piece_count,
                candidate.kind,
                -candidate.width,
                -candidate.height,
            )
        )
        self._candidate_cache = tuple(candidates)
        return candidates

    def _pieces_for(
        self, gate_len: int, side_len: int, gate_count: int = 2
    ) -> list[Piece] | None:
        """Compatibility helper for a four-piece rectangle."""
        if not all(
            c.MIN_WALL_LENGTH <= length <= c.MAX_FACE_LENGTH
            for length in (gate_len, side_len)
        ):
            return None
        pieces = [
            Piece(PieceType.GATE, gate_len),
            Piece(PieceType.WALL, side_len),
            Piece(
                PieceType.GATE if gate_count == 2 else PieceType.WALL,
                gate_len,
            ),
            Piece(PieceType.WALL, side_len),
        ]
        scratch = self.inventory.copy()
        for piece in pieces:
            if not scratch.take_piece(piece):
                return None
        return pieces

    @staticmethod
    def _candidate_from_four_pieces(pieces: list[Piece]) -> ShapeCandidate:
        faces = tuple((piece,) for piece in pieces)
        return ShapeCandidate(
            width=pieces[0].length,
            height=pieces[1].length,
            faces=faces,
            score_hint=0.0,
        )

    def _construction_at(
        self,
        candidate: ShapeCandidate,
        center: tuple[float, float],
        heading_deg: float,
    ) -> Construction:
        theta = math.radians(heading_deg)
        cos_theta, sin_theta = math.cos(theta), math.sin(theta)
        local_polygon = candidate.polygon
        minx, miny, maxx, maxy = local_polygon.bounds
        local_center = ((minx + maxx) / 2, (miny + maxy) / 2)
        local_start = (
            local_polygon.exterior.coords[0][0] - local_center[0],
            local_polygon.exterior.coords[0][1] - local_center[1],
        )
        start = (
            center[0] + local_start[0] * cos_theta - local_start[1] * sin_theta,
            center[1] + local_start[0] * sin_theta + local_start[1] * cos_theta,
        )
        pieces = []
        connectors = []
        face_connectors = candidate.face_connectors or self._right_connectors(
            len(candidate.faces)
        )
        for face_index, face in enumerate(candidate.faces):
            for piece_index, piece in enumerate(face):
                pieces.append(piece)
                connector = (
                    face_connectors[face_index]
                    if piece_index == 0
                    else Connector(ConnectorType.STRAIGHT)
                )
                connectors.append(connector)
        return Construction(start, heading_deg, pieces, connectors)

    def _rectangle_at(
        self, pieces: list[Piece], center: tuple[float, float], heading_deg: float
    ) -> Construction:
        return self._construction_at(
            self._candidate_from_four_pieces(pieces), center, heading_deg
        )

    def _edge_headings(self) -> tuple[float, ...]:
        if self._edge_headings_cache is not None:
            return self._edge_headings_cache
        headings = []
        points = self.room.get_boundary_points()
        for start, end in zip(points, points[1:] + points[:1]):
            edge_heading = math.degrees(
                math.atan2(end[1] - start[1], end[0] - start[0])
            )
            headings.extend((edge_heading % 180, (edge_heading - 90) % 180))
        self._edge_headings_cache = tuple(
            dict.fromkeys(round(heading, 8) for heading in headings)
        )
        return self._edge_headings_cache

    def _headings(self) -> tuple[float, ...]:
        if self._headings_cache is not None:
            return self._headings_cache
        headings = [*HEADINGS_DEG, *self._edge_headings(), *FINE_HEADINGS_DEG]
        self._headings_cache = tuple(
            dict.fromkeys(round(heading, 8) for heading in headings)
        )
        return self._headings_cache

    def _placement_centers(
        self,
        candidate: ShapeCandidate,
        heading: float,
        *,
        include_base: bool = False,
        include_boundary: bool = False,
        include_grid: bool = False,
    ) -> Iterator[tuple[float, float]]:
        polygon = self.room.polygon
        centers = []
        if include_base:
            centroid = polygon.centroid
            representative = polygon.representative_point()
            centers.extend(
                [(centroid.x, centroid.y), (representative.x, representative.y)]
            )

        if include_grid:
            minx, miny, maxx, maxy = polygon.bounds
            centers.extend(
                (
                    minx + (maxx - minx) * i / (GRID_STEPS - 1),
                    miny + (maxy - miny) * j / (GRID_STEPS - 1),
                )
                for i in range(GRID_STEPS)
                for j in range(GRID_STEPS)
            )

        if include_boundary:
            local_polygon = candidate.polygon
            minx, miny, maxx, maxy = local_polygon.bounds
            local_center = ((minx + maxx) / 2, (miny + maxy) / 2)
            vertices = [
                (x - local_center[0], y - local_center[1])
                for x, y in local_polygon.exterior.coords[:-1]
            ]
            local_anchors = vertices + [
                ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                for a, b in zip(vertices, vertices[1:] + vertices[:1])
            ]
            theta = math.radians(heading)
            cos_theta, sin_theta = math.cos(theta), math.sin(theta)
            rotated_anchors = [
                (
                    x * cos_theta - y * sin_theta,
                    x * sin_theta + y * cos_theta,
                )
                for x, y in local_anchors
            ]
            room_points = self.room.get_boundary_points()
            room_points += [
                ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                for a, b in zip(room_points, room_points[1:] + room_points[:1])
            ]
            centers.extend(
                (room_x - anchor_x, room_y - anchor_y)
                for room_x, room_y in room_points
                for anchor_x, anchor_y in rotated_anchors
            )

        seen = set()
        for center in centers:
            key = (round(center[0], 8), round(center[1], 8))
            if key not in seen:
                seen.add(key)
                yield center

    def _find_placement(
        self, candidate: ShapeCandidate | list[Piece]
    ) -> Construction | None:
        if isinstance(candidate, list):
            candidate = self._candidate_from_four_pieces(candidate)

        room = prep(self.room.polygon.buffer(c.TOL))
        room_minx, room_miny, room_maxx, room_maxy = self.room.polygon.bounds
        local_polygon = candidate.polygon
        minx, miny, maxx, maxy = local_polygon.bounds
        local_shape = translate(
            local_polygon,
            xoff=-(minx + maxx) / 2,
            yoff=-(miny + maxy) / 2,
        )
        stages = (
            (self._headings(), {"include_base": True}),
            (self._edge_headings(), {"include_boundary": True}),
            (HEADINGS_DEG, {"include_grid": True}),
        )
        for headings, center_options in stages:
            for heading in headings:
                rotated_shape = rotate(
                    local_shape, heading, origin=(0, 0), use_radians=False
                )
                relative_points = tuple(rotated_shape.exterior.coords[:-1])
                minx, miny, maxx, maxy = rotated_shape.bounds
                for center in self._placement_centers(
                    candidate, heading, **center_options
                ):
                    if (
                        center[0] + minx < room_minx - c.TOL
                        or center[0] + maxx > room_maxx + c.TOL
                        or center[1] + miny < room_miny - c.TOL
                        or center[1] + maxy > room_maxy + c.TOL
                    ):
                        continue
                    placed_shape = Polygon(
                        (x + center[0], y + center[1]) for x, y in relative_points
                    )
                    if not room.contains(placed_shape):
                        continue
                    construction = self._construction_at(candidate, center, heading)
                    if validate_construction(
                        construction, self.room, self.inventory
                    ).valid:
                        return construction
        return None

    def _evaluate_candidates(
        self, max_shapes: int | None = None
    ) -> Iterator[tuple[ShapeCandidate, Construction | None, ValidationResult]]:
        if max_shapes is not None and max_shapes < 1:
            raise ValueError("max_shapes must be positive")
        self.note = None
        if (
            self.inventory.connectors.get(ConnectorType.RIGHT, 0) < 4
            and self.inventory.connectors.get(ConnectorType.DIAGONAL, 0) < 8
        ):
            self.lacks_connectors(ConnectorType.RIGHT, 4)
            return
        candidates = self._candidate_shapes()
        if not candidates:
            self.note = "no inventory-feasible supported shapes with face lengths 5..30"
        if max_shapes is not None:
            candidates = candidates[:max_shapes]
        for candidate in candidates:
            construction, result = self._evaluate_candidate(candidate)
            yield candidate, construction, result

    def _evaluate_candidate(
        self, candidate: ShapeCandidate
    ) -> tuple[Construction | None, ValidationResult]:
        construction = self._find_placement(candidate)
        result = (
            validate_construction(construction, self.room, self.inventory)
            if construction is not None
            else ValidationResult(
                valid=False,
                reason="no valid placement in boundary/grid heading search",
            )
        )
        return construction, result

    def evaluate_shape_candidates(self, max_shapes: int | None = None) -> list[dict]:
        rows = []
        for candidate, construction, result in self._evaluate_candidates(max_shapes):
            rows.append(
                {
                    "shape": candidate.name,
                    "valid": result.valid,
                    "area": result.area,
                    "perimeter": result.perimeter,
                    "score": score_construction(result, self.weights)
                    if construction is not None
                    else 0.0,
                    "reason": result.reason,
                }
            )
        rows.sort(key=lambda row: (row["valid"], row["score"]), reverse=True)
        return rows

    def select_best_candidate(
        self, max_shapes: int | None = None
    ) -> tuple[ShapeCandidate, Construction, ValidationResult, float] | None:
        if max_shapes is not None and max_shapes < 1:
            raise ValueError("max_shapes must be positive")
        self.note = None
        if (
            self.inventory.connectors.get(ConnectorType.RIGHT, 0) < 4
            and self.inventory.connectors.get(ConnectorType.DIAGONAL, 0) < 8
        ):
            self.lacks_connectors(ConnectorType.RIGHT, 4)
            return None
        candidates = self._candidate_shapes()
        if max_shapes is not None:
            candidates = candidates[:max_shapes]
        if not candidates:
            self.note = "no inventory-feasible supported shapes with face lengths 5..30"
            return None

        attempted = 0
        for candidate in candidates:
            if candidate.score_hint <= 0:
                self.note = (
                    f"best remaining shape {candidate.name} has projected score "
                    f"{candidate.score_hint:.2f}; skipped because returning no "
                    "construction scores 0.00"
                )
                return None
            attempted += 1
            construction, result = self._evaluate_candidate(candidate)
            if result.valid and construction is not None:
                score = score_construction(result, self.weights)
                return candidate, construction, result, score
        self.note = (
            f"{attempted} positive-scoring inventory-feasible shapes tried; "
            "none fit the room in the boundary/grid heading search"
        )
        return None

    def build_enclosure(self) -> Construction | None:
        selection = self.select_best_candidate()
        return selection[1] if selection is not None else None
