"""Gymnasium environment for learning Player 10 shape decisions.

An episode has three decisions: choose an inventory-feasible shape (or
opt out), choose its heading, and choose its center.  The completed proposal
is always checked by the simulator's validator and scorer.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from shapely.affinity import rotate
from shapely.geometry import Polygon, box

import src.constants as c
from players.player10.player import GRID_STEPS, Player10, ShapeCandidate
from players.player10.tests.pipeline import iter_scenario_files
from src.enclosure import score_construction, validate_construction, walk_points
from src.scenario import Scenario, generate_scenario, read_scenario
from src.weights import Weights


@dataclass(frozen=True)
class TrainingCase:
    """One immutable scenario and its precomputed rectangle candidates."""

    name: str
    scenario: Scenario
    candidates: tuple[ShapeCandidate, ...]


def prepare_case(
    name: str,
    scenario: Scenario,
    max_candidates: int | None = None,
    candidate_kinds: set[str] | None = None,
) -> TrainingCase:
    player = Player10(scenario.room, scenario.inventory, scenario.weights)
    candidates = tuple(player._candidate_shapes())
    if candidate_kinds is not None:
        candidates = tuple(
            candidate for candidate in candidates if candidate.kind in candidate_kinds
        )
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    return TrainingCase(name=name, scenario=scenario, candidates=candidates)


def load_training_cases(
    scenario_paths: Iterable[str | Path] | None = None,
    *,
    max_candidates: int | None = None,
    generated_count: int = 0,
    seed: int = 10,
    progress: Callable[[str], None] | None = None,
    candidate_kinds: set[str] | None = None,
) -> list[TrainingCase]:
    """Load bundled scenarios and optionally add deterministic generated ones.

    Candidate generation is done once here because it is much more expensive
    than an environment step.
    """
    if max_candidates is not None and max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    paths = (
        [Path(path) for path in scenario_paths]
        if scenario_paths is not None
        else iter_scenario_files()
    )
    cases = []
    for index, path in enumerate(paths, start=1):
        if progress is not None:
            progress(f"preparing scenario {index}/{len(paths)}: {path}")
        cases.append(
            prepare_case(
                path.as_posix(),
                read_scenario(str(path)),
                max_candidates,
                candidate_kinds,
            )
        )

    rng = random.Random(seed)
    area_weights = (1.0, 1.5, 2.0, 3.0, 4.0, 10.0, 20.0, 40.0)
    perimeter_weights = (0.0, -1.0, -2.0, -3.0, -5.0, -8.0, -10.0, -24.0, -70.0)
    gate_weights = (0.0, 50.0, 75.0, 100.0, 150.0, 200.0, 400.0, 500.0, 1000.0)
    for offset in range(generated_count):
        scenario_seed = seed + offset
        if progress is not None:
            progress(
                f"preparing generated scenario {offset + 1}/{generated_count}: "
                f"seed {scenario_seed}"
            )
        scenario = generate_scenario(scenario_seed, difficulty="scarce")
        scenario = Scenario(
            room=scenario.room,
            inventory=scenario.inventory,
            weights=Weights(
                A=rng.choice(area_weights),
                C=rng.choice(perimeter_weights),
                G=rng.choice(gate_weights),
            ),
        )
        cases.append(
            prepare_case(
                f"generated/seed_{scenario_seed}.json",
                scenario,
                max_candidates,
                candidate_kinds,
            )
        )
    if not cases:
        raise ValueError("at least one training scenario is required")
    return cases


class PlaypenRectangleEnv(gym.Env[np.ndarray, int]):
    """Three-step, masked-action environment backed by simulator validation."""

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["ansi"],
        "render_fps": 1,
    }

    def __init__(
        self,
        cases: list[TrainingCase],
        *,
        max_candidates: int | None = None,
        max_headings: int | None = None,
        max_centers: int | None = None,
        reward_scale: float = 1000.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if not cases:
            raise ValueError("cases cannot be empty")
        if reward_scale <= 0:
            raise ValueError("reward_scale must be positive")

        inferred_candidates = max(1, max(len(case.candidates) for case in cases))
        inferred_headings = max(
            1,
            max(
                len(
                    Player10(
                        case.scenario.room,
                        case.scenario.inventory,
                        case.scenario.weights,
                    )._headings()
                )
                for case in cases
            ),
        )
        # Every placement list contains two base centers, GRID_STEPS squared
        # grid centers, and vertex/edge-midpoint anchor pairs for both the
        # room and candidate polygon.
        inferred_centers = max(
            1,
            max(
                2
                + GRID_STEPS**2
                + 4
                * len(case.scenario.room.get_boundary_points())
                * max(
                    (len(candidate.faces) for candidate in case.candidates), default=4
                )
                for case in cases
            ),
        )

        def capacity(requested: int | None, inferred: int, name: str) -> int:
            if requested is None:
                return inferred
            if requested < 1:
                raise ValueError(f"{name} must be positive")
            return requested

        max_candidates = capacity(max_candidates, inferred_candidates, "max_candidates")
        max_headings = capacity(max_headings, inferred_headings, "max_headings")
        max_centers = capacity(max_centers, inferred_centers, "max_centers")
        if any(len(case.candidates) > max_candidates for case in cases):
            raise ValueError("a case contains more candidates than max_candidates")

        self.cases = cases
        self.max_candidates = max_candidates
        self.max_headings = max_headings
        self.max_centers = max_centers
        self.reward_scale = reward_scale
        self.render_mode = render_mode
        self.action_size = max(max_candidates + 1, max_headings, max_centers)
        self.action_space = spaces.Discrete(self.action_size)

        # Global/phase features + inventories + candidate, heading, and center tables.
        self._base_size = 3 + 7 + 3 + 52 + 3 + 6 + 3
        self._observation_size = (
            self._base_size + max_candidates * 6 + max_headings * 3 + max_centers * 3
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._observation_size,),
            dtype=np.float32,
        )

        self.case_index = 0
        self.phase = 0
        self.selected_candidate_index: int | None = None
        self.selected_heading_index: int | None = None
        self._player: Player10 | None = None
        self._headings: tuple[float, ...] = ()
        self._centers: tuple[tuple[float, float], ...] = ()
        self._last_info: dict[str, Any] = {}

    @property
    def case(self) -> TrainingCase:
        return self.cases[self.case_index]

    @property
    def player(self) -> Player10:
        if self._player is None:
            raise RuntimeError("reset() must be called before using the environment")
        return self._player

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        requested = None if options is None else options.get("case_index")
        if requested is None:
            self.case_index = int(self.np_random.integers(len(self.cases)))
        else:
            self.case_index = int(requested)
            if not 0 <= self.case_index < len(self.cases):
                raise IndexError("case_index is out of range")
        scenario = self.case.scenario
        self._player = Player10(scenario.room, scenario.inventory, scenario.weights)
        self.phase = 0
        self.selected_candidate_index = None
        self.selected_heading_index = None
        self._headings = tuple(self.player._headings()[: self.max_headings])
        self._centers = ()
        self._last_info = {"scenario": self.case.name, "phase": self.phase}
        return self._get_observation(), dict(self._last_info)

    def _selected_candidate(self) -> ShapeCandidate | None:
        if self.selected_candidate_index is None:
            return None
        return self.case.candidates[self.selected_candidate_index]

    def _selected_heading(self) -> float | None:
        if self.selected_heading_index is None:
            return None
        return self._headings[self.selected_heading_index]

    def _set_centers(self) -> None:
        candidate = self._selected_candidate()
        heading = self._selected_heading()
        if candidate is None or heading is None:
            self._centers = ()
            return
        self._centers = tuple(
            self.player._placement_centers(
                candidate,
                heading,
                include_base=True,
                include_boundary=True,
                include_grid=True,
            )
        )[: self.max_centers]

    @staticmethod
    def _candidate_features(candidate: ShapeCandidate) -> tuple[float, ...]:
        return (
            candidate.width / c.MAX_FACE_LENGTH,
            candidate.height / c.MAX_FACE_LENGTH,
            float(np.clip(candidate.score_hint / 20_000.0, -1.0, 1.0)),
            min(candidate.gate_count / 8.0, 1.0),
            min(candidate.piece_count / 20.0, 1.0),
            1.0,
        )

    def _get_observation(self) -> np.ndarray:
        scenario = self.case.scenario
        room = scenario.room.polygon
        minx, miny, maxx, maxy = room.bounds
        width = max(maxx - minx, 1e-9)
        height = max(maxy - miny, 1e-9)
        centroid = room.centroid
        phase = [1.0 if self.phase == i else 0.0 for i in range(3)]
        room_features = [
            min(width / 50.0, 1.0),
            min(height / 50.0, 1.0),
            room.area / (width * height),
            min(room.length / 300.0, 1.0),
            (centroid.x - minx) / width,
            (centroid.y - miny) / height,
            min(len(scenario.room.get_boundary_points()) / 64.0, 1.0),
        ]
        weight_features = [
            min(scenario.weights.A / 40.0, 1.0),
            max(scenario.weights.C / 70.0, -1.0),
            min(scenario.weights.G / 1000.0, 1.0),
        ]
        inventory_features = []
        for pool in (scenario.inventory.walls, scenario.inventory.gates):
            inventory_features.extend(
                min(pool.get(length, 0) / 10.0, 1.0)
                for length in range(c.MIN_WALL_LENGTH, c.MAX_FACE_LENGTH + 1)
            )
        inventory_features.extend(
            min(scenario.inventory.connectors.get(kind, 0) / 40.0, 1.0)
            for kind in self._connector_order()
        )

        selected_candidate = self._selected_candidate()
        selected_features = (
            list(self._candidate_features(selected_candidate))
            if selected_candidate is not None
            else [0.0] * 6
        )
        selected_heading = self._selected_heading()
        selected_heading_features = (
            [
                math.sin(math.radians(selected_heading)),
                math.cos(math.radians(selected_heading)),
                1.0,
            ]
            if selected_heading is not None
            else [0.0] * 3
        )

        candidate_table = np.zeros((self.max_candidates, 6), dtype=np.float32)
        for index, candidate in enumerate(self.case.candidates):
            candidate_table[index] = self._candidate_features(candidate)
        heading_table = np.zeros((self.max_headings, 3), dtype=np.float32)
        for index, heading in enumerate(self._headings):
            heading_table[index] = (
                math.sin(math.radians(heading)),
                math.cos(math.radians(heading)),
                1.0,
            )
        center_table = np.zeros((self.max_centers, 3), dtype=np.float32)
        for index, (x, y) in enumerate(self._centers):
            center_table[index] = (
                float(np.clip(2 * (x - minx) / width - 1, -1, 1)),
                float(np.clip(2 * (y - miny) / height - 1, -1, 1)),
                1.0,
            )

        observation = np.concatenate(
            (
                np.asarray(
                    phase
                    + room_features
                    + weight_features
                    + inventory_features
                    + selected_features
                    + selected_heading_features,
                    dtype=np.float32,
                ),
                candidate_table.ravel(),
                heading_table.ravel(),
                center_table.ravel(),
            )
        )
        if observation.shape != self.observation_space.shape:
            raise RuntimeError(
                f"observation shape {observation.shape} does not match "
                f"{self.observation_space.shape}"
            )
        return observation

    @staticmethod
    def _connector_order():
        from src.pieces import ConnectorType

        return (
            ConnectorType.STRAIGHT,
            ConnectorType.RIGHT,
            ConnectorType.DIAGONAL,
        )

    def action_masks(self) -> np.ndarray:
        """Mask actions unavailable in the current decision phase."""
        mask = np.zeros(self.action_size, dtype=bool)
        if self.phase == 0:
            mask[: len(self.case.candidates) + 1] = True  # action 0 opts out
        elif self.phase == 1:
            mask[: len(self._headings)] = True
        elif self.phase == 2:
            candidate = self._selected_candidate()
            heading = self._selected_heading()
            if candidate is None or heading is None:
                raise RuntimeError("candidate and heading must precede center choice")
            local = rotate(
                box(
                    -candidate.width / 2,
                    -candidate.height / 2,
                    candidate.width / 2,
                    candidate.height / 2,
                ),
                heading,
                origin=(0, 0),
                use_radians=False,
            )
            lminx, lminy, lmaxx, lmaxy = local.bounds
            rminx, rminy, rmaxx, rmaxy = self.case.scenario.room.polygon.bounds
            for index, (x, y) in enumerate(self._centers):
                mask[index] = (
                    x + lminx >= rminx - c.TOL
                    and x + lmaxx <= rmaxx + c.TOL
                    and y + lminy >= rminy - c.TOL
                    and y + lmaxy <= rmaxy + c.TOL
                )
            if not mask.any():
                mask[: len(self._centers)] = True
        return mask

    def step(self, action: int):
        action = int(action)
        mask = self.action_masks()
        if action < 0 or action >= self.action_size or not mask[action]:
            info = {
                "terminal": True,
                "scenario": self.case.name,
                "shape": "none",
                "candidate_index": self.selected_candidate_index,
                "heading": self._selected_heading(),
                "center": None,
                "valid": False,
                "opted_out": False,
                "simulator_score": float(c.INVALID_SCORE),
                "area": 0.0,
                "perimeter": 0.0,
                "reason": f"action {action} is unavailable in phase {self.phase}",
                "masked_action": True,
            }
            self._last_info = info
            return self._get_observation(), -1.0, True, False, info

        if self.phase == 0:
            if action == 0:
                info = {
                    "terminal": True,
                    "scenario": self.case.name,
                    "shape": "none",
                    "candidate_index": None,
                    "heading": None,
                    "center": None,
                    "valid": False,
                    "opted_out": True,
                    "simulator_score": 0.0,
                    "area": 0.0,
                    "perimeter": 0.0,
                    "reason": "agent opted out",
                }
                self._last_info = info
                return self._get_observation(), 0.0, True, False, info
            self.selected_candidate_index = action - 1
            self.phase = 1
            info = {"scenario": self.case.name, "phase": self.phase}
            return self._get_observation(), 0.0, False, False, info

        if self.phase == 1:
            self.selected_heading_index = action
            self.phase = 2
            self._set_centers()
            info = {"scenario": self.case.name, "phase": self.phase}
            return self._get_observation(), 0.0, False, False, info

        candidate = self._selected_candidate()
        heading = self._selected_heading()
        if candidate is None or heading is None:
            raise RuntimeError("invalid environment state")
        center = self._centers[action]
        construction = self.player._construction_at(candidate, center, heading)
        result = validate_construction(
            construction, self.case.scenario.room, self.case.scenario.inventory
        )
        simulator_score = score_construction(result, self.case.scenario.weights)
        if result.valid:
            reward = simulator_score / self.reward_scale
            inside_fraction = 1.0
        else:
            polygon = Polygon(walk_points(construction)[:-1])
            inside_fraction = (
                polygon.intersection(self.case.scenario.room.polygon).area
                / polygon.area
                if polygon.is_valid and polygon.area > 0
                else 0.0
            )
            # Preserve the simulator's -1 scaled invalid reward and add a
            # bounded geometric hint so the policy can learn where to place.
            reward = -1.0 + 0.9 * inside_fraction
        info = {
            "terminal": True,
            "scenario": self.case.name,
            "shape": candidate.name,
            "candidate_index": self.selected_candidate_index,
            "heading": round(heading, 8),
            "center": [round(center[0], 8), round(center[1], 8)],
            "valid": result.valid,
            "opted_out": False,
            "simulator_score": float(simulator_score),
            "area": float(result.area),
            "perimeter": float(result.perimeter),
            "inside_fraction": float(inside_fraction),
            "reason": result.reason,
        }
        self._last_info = info
        return self._get_observation(), float(reward), True, False, info

    def render(self) -> str | None:
        text = (
            f"{self.case.name}: phase={self.phase}, "
            f"candidate={self.selected_candidate_index}, "
            f"heading={self._selected_heading()}, result={self._last_info}"
        )
        if self.render_mode == "ansi":
            return text
        print(text)
        return None
