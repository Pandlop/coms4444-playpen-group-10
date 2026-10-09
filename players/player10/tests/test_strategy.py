"""Player 10 regressions: uv run python -m unittest players.player10.tests.test_strategy."""

import contextlib
import csv
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shapely.affinity import rotate
from shapely.geometry import Polygon, box

from players.player10.player import Player10
from players.player10.tests.pipeline import (
    _format_table,
    best_shape_for_scenario,
    export_results,
    iter_scenario_files,
    main,
    run_shape_pipeline,
)
from src.enclosure import score_construction, validate_construction
from src.inventory import Inventory
from src.pieces import ConnectorType
from src.room import Room
from src.scenario import Scenario, read_scenario
from src.weights import Weights


def make_player(walls=None, gates=None, right=4, straight=0, room=None, weights=None):
    return Player10(
        Room(room if room is not None else box(0, 0, 40, 40)),
        Inventory(
            walls=walls if walls is not None else {5: 2, 10: 2},
            gates=gates if gates is not None else {5: 2},
            connectors={
                ConnectorType.RIGHT: right,
                ConnectorType.STRAIGHT: straight,
            },
        ),
        weights if weights is not None else Weights(A=2, C=-3, G=100),
    )


class StrategyTests(unittest.TestCase):
    def test_piece_limits_two_gates_and_inventory_preservation(self):
        player = make_player(walls={4: 2, 5: 2, 31: 2}, gates={4: 2, 5: 2, 31: 2})
        before = player.inventory.to_dict()
        self.assertEqual(
            [candidate.name for candidate in player._candidate_shapes()],
            ["5x5 (2 gates, 4 pieces)"],
        )
        construction = player.build_enclosure()
        self.assertIsNotNone(construction)
        self.assertEqual(sum(piece.is_gate for piece in construction.pieces), 2)
        result = validate_construction(construction, player.room, player.inventory)
        self.assertTrue(result.valid, result.reason)
        self.assertTrue(result.second_gate_bonus)
        self.assertEqual(score_construction(result, player.weights), 1090)
        self.assertEqual(player.inventory.to_dict(), before)

    def test_counts_and_gate_wall_orientations(self):
        player = make_player(walls={5: 1, 10: 2}, gates={5: 2, 10: 2})
        self.assertEqual(
            [candidate.name for candidate in player._candidate_shapes()],
            ["10x10 (2 gates, 4 pieces)", "10x5 (2 gates, 4 pieces)"],
        )
        fallback = make_player(gates={5: 1}).build_enclosure()
        self.assertIsNotNone(fallback)
        self.assertEqual(sum(piece.is_gate for piece in fallback.pieces), 1)
        self.assertIsNone(make_player(walls={5: 1}).build_enclosure())

    def test_missing_connectors_prevents_placement_search(self):
        player = make_player(right=3)
        with patch.object(
            player, "_candidate_shapes", side_effect=AssertionError("must not search")
        ):
            self.assertIsNone(player.build_enclosure())
            self.assertEqual(player.evaluate_shape_candidates(), [])
        self.assertIn("Not enough right connectors", player.note)

    def test_best_score_can_be_smaller_than_first_valid_rectangle(self):
        player = make_player(weights=Weights(A=1, C=-10, G=0))
        self.assertEqual(
            [candidate.name for candidate in player._candidate_shapes()],
            ["5x5 (2 gates, 4 pieces)", "10x5 (1 gate, 4 pieces)"],
        )
        rows = player.evaluate_shape_candidates()
        self.assertEqual(rows[0]["shape"], "5x5 (2 gates, 4 pieces)")
        self.assertEqual(
            player.evaluate_shape_candidates(max_shapes=1)[0]["shape"],
            "5x5 (2 gates, 4 pieces)",
        )
        construction = player.build_enclosure()
        result = validate_construction(construction, player.room, player.inventory)
        self.assertTrue(result.valid)
        self.assertEqual(score_construction(result, player.weights), rows[0]["score"])

    def test_valid_nonpositive_score_is_skipped_for_zero_point_opt_out(self):
        player = make_player(weights=Weights(A=1, C=-1000, G=0))
        self.assertIsNone(player.build_enclosure())
        self.assertIn(
            "skipped because returning no construction scores 0.00", player.note
        )
        self.assertLess(player._candidate_shapes()[0].score_hint, -1000)

    def test_heading_sweep_finds_rotated_rectangle(self):
        player = make_player(
            walls={5: 2}, gates={10: 2}, room=rotate(box(0, 0, 12, 7), 15)
        )
        construction = player.build_enclosure()
        self.assertIsNotNone(construction)
        self.assertEqual(construction.start_heading, 15)
        self.assertTrue(
            validate_construction(construction, player.room, player.inventory).valid
        )

    def test_grid_finds_rectangle_away_from_concave_room_centroid(self):
        player = make_player(
            walls={5: 2},
            room=Polygon([(0, 0), (30, 0), (30, 10), (10, 10), (10, 30), (0, 30)]),
        )
        pieces = player._pieces_for(5, 5)
        centroid = player.room.polygon.centroid
        centered = player._rectangle_at(pieces, (centroid.x, centroid.y), 0)
        self.assertFalse(
            validate_construction(centered, player.room, player.inventory).valid
        )
        construction = player.build_enclosure()
        self.assertIsNotNone(construction)
        self.assertTrue(
            validate_construction(construction, player.room, player.inventory).valid
        )

    def test_no_placement_returns_none_with_zero_pipeline_score(self):
        player = make_player(room=box(0, 0, 4, 4))
        self.assertIsNone(player.build_enclosure())
        rows = player.evaluate_shape_candidates()
        self.assertTrue(all(not row["valid"] and row["score"] == 0 for row in rows))

    def test_deterministic_inventory_order_and_score_ties(self):
        first = make_player(
            walls={5: 2, 10: 2}, gates={5: 2, 10: 2}, weights=Weights(A=0, C=0, G=0)
        )
        second = make_player(
            walls={10: 2, 5: 2}, gates={10: 2, 5: 2}, weights=first.weights
        )
        self.assertEqual(first._candidate_shapes(), second._candidate_shapes())
        self.assertEqual(first.build_enclosure(), second.build_enclosure())
        self.assertEqual(first.build_enclosure().pieces[0].length, 10)

    def test_smaller_rectangle_tried_when_largest_does_not_fit(self):
        player = make_player(room=box(0, 0, 7, 7))
        self.assertFalse(player.evaluate_shape_candidates(max_shapes=1)[0]["valid"])
        construction = player.build_enclosure()
        self.assertIsNotNone(construction)
        result = validate_construction(construction, player.room, player.inventory)
        self.assertTrue(result.valid)
        self.assertEqual(result.area, 25)

    def test_maximum_face_length_is_allowed(self):
        player = make_player(walls={30: 2}, gates={30: 2})
        construction = player.build_enclosure()
        self.assertTrue(
            validate_construction(construction, player.room, player.inventory).valid
        )

    def test_different_gate_lengths_can_form_adjacent_faces(self):
        player = make_player(
            walls={8: 1, 10: 1},
            gates={8: 1, 10: 1},
            room=Polygon([(0, 0), (30, 0), (24, 24), (6, 24)]),
            weights=Weights(A=1, C=-10, G=30),
        )
        construction = player.build_enclosure()
        result = validate_construction(construction, player.room, player.inventory)
        self.assertTrue(result.valid, result.reason)
        self.assertTrue(result.second_gate_bonus)
        self.assertEqual(score_construction(result, player.weights), 750)

    def test_composed_faces_and_boundary_anchors_find_exact_fit(self):
        scenario = read_scenario("scenarios/players/player10/i_room.json")
        player = Player10(scenario.room, scenario.inventory, scenario.weights)
        construction = player.build_enclosure()
        result = validate_construction(construction, player.room, player.inventory)
        self.assertTrue(result.valid, result.reason)
        self.assertEqual(result.area, 676)
        self.assertEqual(score_construction(result, player.weights), 2040)
        self.assertGreater(len(construction.pieces), 4)

    def test_l_shape_beats_rectangle_and_uses_reflex_corner(self):
        room = Polygon([(0, 0), (20, 0), (20, 10), (10, 10), (10, 20), (0, 20)])
        player = make_player(
            walls={10: 4, 20: 1},
            gates={20: 1},
            right=6,
            room=room,
            weights=Weights(A=10, C=-1, G=0),
        )
        selection = player.select_best_candidate()
        self.assertIsNotNone(selection)
        candidate, construction, result, score = selection
        self.assertEqual(candidate.kind, "L")
        self.assertTrue(result.valid, result.reason)
        self.assertEqual(result.area, 300)
        self.assertEqual(score, 3920)
        self.assertTrue(any(connector.reflex for connector in construction.connectors))

    def test_diagonal_octagon_can_build_without_right_connectors(self):
        x = y = 0.0
        points = []
        for angle in range(0, 360, 45):
            points.append((x, y))
            x += 10 * math.cos(math.radians(angle))
            y += 10 * math.sin(math.radians(angle))
        player = Player10(
            Room(Polygon(points)),
            Inventory(
                walls={10: 7},
                gates={10: 1},
                connectors={ConnectorType.DIAGONAL: 8},
            ),
            Weights(A=2, C=-3, G=0),
        )
        selection = player.select_best_candidate()
        self.assertIsNotNone(selection)
        candidate, construction, result, _ = selection
        self.assertEqual(candidate.kind, "octagon")
        self.assertTrue(result.valid, result.reason)
        self.assertTrue(
            all(
                connector.connector_type == ConnectorType.DIAGONAL
                for connector in construction.connectors
            )
        )

    def test_u_shape_uses_two_reflex_corners(self):
        room = Polygon(
            [
                (0, 0),
                (30, 0),
                (30, 20),
                (25, 20),
                (25, 10),
                (5, 10),
                (5, 20),
                (0, 20),
            ]
        )
        player = make_player(
            walls={5: 2, 10: 2, 20: 3},
            gates={30: 1},
            right=8,
            room=room,
            weights=Weights(A=10, C=-1, G=0),
        )
        selection = player.select_best_candidate()
        self.assertIsNotNone(selection)
        candidate, construction, result, score = selection
        self.assertEqual(candidate.kind, "U")
        self.assertTrue(result.valid, result.reason)
        self.assertAlmostEqual(result.area, 400)
        self.assertAlmostEqual(score, 4880)
        self.assertEqual(
            sum(connector.reflex for connector in construction.connectors), 2
        )

    def test_general_templates_improve_bundled_rectangle_baselines(self):
        cases = (
            ("scenarios/players/player9/crown.json", "octagon", 9500),
            ("scenarios/players/player6/spiral.json", "octagon", 2990),
            ("scenarios/players/player4/irregular_room_second_gate.json", "L", 1866),
            ("scenarios/players/player5/rectangle_room.json", "L", 1974),
        )
        for path, kind, old_score in cases:
            with self.subTest(path=path):
                scenario = read_scenario(path)
                selection = Player10(
                    scenario.room, scenario.inventory, scenario.weights
                ).select_best_candidate()
                self.assertIsNotNone(selection)
                candidate, _, result, score = selection
                self.assertEqual(candidate.kind, kind)
                self.assertTrue(result.valid, result.reason)
                self.assertGreater(score, old_score)

    def test_nonpositive_caps_rejected(self):
        for cap in (0, -1):
            with self.assertRaises(ValueError):
                make_player().evaluate_shape_candidates(cap)
            with self.assertRaises(ValueError):
                run_shape_pipeline(max_shapes=cap)


class PipelineTests(unittest.TestCase):
    def test_pipeline_reports_the_strategys_negative_score_opt_out(self):
        row = best_shape_for_scenario("scenarios/players/player10/zig_room.json")
        self.assertFalse(row["valid"])
        self.assertEqual(row["shape"], "none")
        self.assertEqual(row["score"], 0.0)
        self.assertIn("has projected score", row["reason"])
        self.assertIn("returning no construction scores 0.00", row["reason"])

    def test_csv_and_json_exports_preserve_order_fields_and_reasons(self):
        rows = [
            {
                "scenario": "scenarios/players/player10/zig_room.json",
                "shape": "5x5",
                "valid": True,
                "area": 25.004,
                "perimeter": 20.006,
                "score": -50.006,
                "reason": "valid",
            },
            {
                "scenario": "scenarios/missing.json",
                "shape": "none",
                "valid": False,
                "area": 0.0,
                "perimeter": 0.0,
                "score": 0.0,
                "reason": 'missing pieces, including "gates"\nfull reason preserved',
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = export_results(rows, Path(tmp) / "nested/results.csv")
            with csv_path.open(newline="", encoding="utf-8") as stream:
                exported_csv = list(csv.DictReader(stream))
            self.assertEqual(
                [row["scenario"] for row in exported_csv],
                [row["scenario"] for row in rows],
            )
            self.assertEqual(exported_csv[0]["area"], "25.00")
            self.assertEqual(exported_csv[0]["perimeter"], "20.01")
            self.assertEqual(exported_csv[0]["score"], "-50.01")
            self.assertEqual(exported_csv[1]["valid"], "false")
            self.assertEqual(exported_csv[1]["reason"], rows[1]["reason"])
            json_path = export_results(rows, Path(tmp) / "results.json")
            exported_json = json.loads(json_path.read_text())
            self.assertEqual(exported_json[0]["area"], 25.0)
            self.assertEqual(exported_json[0]["perimeter"], 20.01)
            self.assertEqual(exported_json[0]["score"], -50.01)
            self.assertEqual(exported_json[1], rows[1])
            self.assertIn("\n  {\n", json_path.read_text())
            with self.assertRaises(ValueError):
                export_results(rows, Path(tmp) / "results.txt")

    def test_readable_table_wraps_reasons_and_aligns_numbers(self):
        row = {
            "scenario": "scenarios/players/player10/zig_room.json",
            "shape": "5x5",
            "valid": True,
            "area": 25.0,
            "perimeter": 20.0,
            "score": 12345.0,
            "reason": "long detailed explanation " * 12,
        }
        table = _format_table([row])
        self.assertIn("player10/zig_room.json", table)
        self.assertIn("5 × 5", table)
        self.assertIn("12,345.00", table)
        self.assertGreater(len(table.splitlines()), 3)
        self.assertLessEqual(max(map(len, table.splitlines())), 160)

    def test_recursive_discovery_both_roots_and_sorted_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [
                root / "scenarios/a.json",
                root / "scenarios/players/10/deep/a.json",
                root / "players/10/a.json",
            ]
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
            self.assertEqual(
                iter_scenario_files(root), sorted(path.resolve() for path in paths)
            )
            self.assertEqual(iter_scenario_files(root / "absent"), [])

    def test_sweep_reports_errors_and_sorts_valid_before_no_solution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / name for name in ("bad.json", "none.json", "valid.json")]
            paths[0].write_text("{broken")
            for path, player in zip(
                paths[1:],
                [make_player(right=0), make_player()],
            ):
                path.write_text(
                    json.dumps(
                        Scenario(
                            player.room, player.inventory, player.weights
                        ).to_dict()
                    )
                )
            with (
                patch(
                    "players.player10.tests.pipeline.iter_scenario_files",
                    return_value=paths,
                ),
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                rows = run_shape_pipeline()
            self.assertTrue(rows[0]["valid"])
            self.assertGreater(rows[0]["score"], 0)
            self.assertIn("scenario error:", rows[1]["reason"])
            self.assertIn("Not enough right connectors", rows[2]["reason"])
            self.assertIn("No-solution breakdown:", output.getvalue())
            for header in (
                "scenario",
                "shape",
                "valid",
                "area",
                "perimeter",
                "score",
                "reason",
            ):
                self.assertIn(header, output.getvalue().splitlines()[0])
            self.assertIn(
                "scenario error:",
                best_shape_for_scenario(root / "missing.json")["reason"],
            )

    def test_cli_defaults_to_all_and_passes_cap(self):
        for arguments, cap in (
            ([], None),
            (["--all-scenarios", "--max-shapes", "1"], 1),
        ):
            with (
                patch("sys.argv", ["pipeline", *arguments]),
                patch("players.player10.tests.pipeline.run_shape_pipeline") as run,
            ):
                main()
            run.assert_called_once_with(None, max_shapes=cap, output_path=None)
        with (
            patch("sys.argv", ["pipeline", "--output", "results/player10.json"]),
            patch("players.player10.tests.pipeline.run_shape_pipeline") as run,
        ):
            main()
        run.assert_called_once_with(
            None, max_shapes=None, output_path=Path("results/player10.json")
        )
        with (
            patch("sys.argv", ["pipeline", "--max-shapes", "0"]),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as exc:
                main()
            self.assertEqual(exc.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
