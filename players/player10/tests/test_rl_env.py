"""Archived RL tests; skipped when the optional packages are unavailable."""

import tempfile
import unittest
from pathlib import Path

from shapely.geometry import box

from src.inventory import Inventory
from src.pieces import ConnectorType
from src.room import Room
from src.scenario import Scenario
from src.weights import Weights

try:
    import numpy as np

    from players.player10.archive.rl.rl_env import PlaypenRectangleEnv, prepare_case
    from players.player10.archive.rl.train_rl import _append_csv
except ModuleNotFoundError as exc:
    RL_IMPORT_ERROR = str(exc)
else:
    RL_IMPORT_ERROR = None


def make_case():
    scenario = Scenario(
        room=Room(box(0, 0, 40, 40)),
        inventory=Inventory(
            walls={5: 2},
            gates={5: 2},
            connectors={ConnectorType.RIGHT: 4},
        ),
        weights=Weights(A=2, C=-3, G=100),
    )
    return prepare_case("test.json", scenario, max_candidates=8)


def make_many_candidate_case():
    scenario = Scenario(
        room=Room(box(0, 0, 50, 50)),
        inventory=Inventory(
            walls={5: 4, 10: 4},
            gates={5: 2, 10: 2},
            connectors={
                ConnectorType.RIGHT: 4,
                ConnectorType.STRAIGHT: 12,
            },
        ),
        weights=Weights(A=2, C=-3, G=100),
    )
    return scenario


@unittest.skipIf(
    RL_IMPORT_ERROR is not None, f"optional RL package missing: {RL_IMPORT_ERROR}"
)
class EnvironmentTests(unittest.TestCase):
    def test_default_exposes_every_candidate_and_infers_other_capacities(self):
        scenario = make_many_candidate_case()
        full_case = prepare_case("many.json", scenario)
        capped_case = prepare_case("many.json", scenario, max_candidates=1)
        self.assertGreater(len(full_case.candidates), 1)
        self.assertEqual(len(capped_case.candidates), 1)

        env = PlaypenRectangleEnv([full_case])
        self.assertEqual(env.max_candidates, len(full_case.candidates))
        self.assertGreaterEqual(
            env.max_headings, len(env.player._headings()) if env._player else 1
        )
        self.assertGreaterEqual(
            env.max_centers,
            2 + 9**2 + 16 * len(scenario.room.get_boundary_points()),
        )

    def test_opt_out_is_zero_score_terminal_action(self):
        env = PlaypenRectangleEnv([make_case()], max_candidates=8)
        observation, info = env.reset(seed=3, options={"case_index": 0})
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(info["scenario"], "test.json")
        self.assertTrue(env.action_masks()[0])
        _, reward, terminated, truncated, info = env.step(0)
        self.assertEqual(reward, 0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["opted_out"])
        self.assertEqual(info["simulator_score"], 0)

    def test_centered_rectangle_uses_exact_simulator_score(self):
        env = PlaypenRectangleEnv([make_case()], max_candidates=8)
        env.reset(seed=3, options={"case_index": 0})
        _, reward, terminated, _, _ = env.step(1)  # first candidate
        self.assertEqual(reward, 0)
        self.assertFalse(terminated)
        self.assertEqual(env.phase, 1)
        env.step(0)  # zero-degree heading
        self.assertEqual(env.phase, 2)
        self.assertTrue(env.action_masks()[0])
        observation, reward, terminated, truncated, info = env.step(0)  # centroid
        self.assertTrue(env.observation_space.contains(observation))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["valid"], info["reason"])
        self.assertEqual(info["simulator_score"], 1090)
        self.assertAlmostEqual(reward, 1.09)

    def test_masked_action_terminates_safely_for_gym_compatibility(self):
        env = PlaypenRectangleEnv([make_case()], max_candidates=8)
        env.reset(options={"case_index": 0})
        masked = int(np.flatnonzero(~env.action_masks())[0])
        _, reward, terminated, _, info = env.step(masked)
        self.assertEqual(reward, -1)
        self.assertTrue(terminated)
        self.assertTrue(info["masked_action"])
        self.assertEqual(info["simulator_score"], -1000)

    def test_progress_csv_appends_one_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.csv"
            _append_csv(path, ["step", "score"], {"step": 1, "score": 10})
            _append_csv(path, ["step", "score"], {"step": 2, "score": 20})
            self.assertEqual(
                path.read_text().splitlines(), ["step,score", "1,10", "2,20"]
            )


if __name__ == "__main__":
    unittest.main()
