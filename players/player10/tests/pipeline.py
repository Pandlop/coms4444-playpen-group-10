"""Evaluate Player 10 on one scenario or every bundled JSON scenario."""

import argparse
import csv
import json
import textwrap
from collections import Counter
from pathlib import Path

from shapely.errors import GEOSException

from players.player10.player import Player10
from src.room import InvalidRoomException
from src.scenario import read_scenario

ROOT = Path(__file__).resolve().parents[3]
RESULT_FIELDS = ["scenario", "shape", "valid", "area", "perimeter", "score", "reason"]


def iter_scenario_files(base_dir: str | Path | None = None) -> list[Path]:
    root = Path(base_dir) if base_dir is not None else ROOT
    # scenarios/players is covered recursively; also include JSONs in players/.
    files = {
        path.resolve()
        for folder in (root / "scenarios", root / "players")
        for path in folder.rglob("*.json")
        if path.is_file()
    }
    return sorted(files)


def _scenario_name(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _format_table(rows: list[dict]) -> str:
    headers = RESULT_FIELDS
    if not rows:
        return "No scenario files found."
    values = [headers]
    for row in rows:
        values.append(
            [
                _scenario_name(row["scenario"])
                .removeprefix("scenarios/")
                .removeprefix("players/"),
                str(row["shape"]).replace("x", " × "),
                "yes" if row["valid"] else "no",
                f"{row['area']:,.2f}",
                f"{row['perimeter']:,.2f}",
                f"{row['score']:,.2f}",
                str(row["reason"]),
            ]
        )
    widths = [max(len(row[i]) for row in values) for i in range(len(headers))]
    widths[0], widths[-1] = min(widths[0], 42), min(widths[-1], 60)

    def format_line(row: list[str]) -> str:
        return " | ".join(
            value.rjust(width) if 3 <= i <= 5 else value.ljust(width)
            for i, (value, width) in enumerate(zip(row, widths))
        ).rstrip()

    lines = [format_line(headers)]
    lines.append(" | ".join("-" * width for width in widths))
    for row in values[1:]:
        cells = [
            textwrap.wrap(value, width=width) or [""]
            for value, width in zip(row, widths)
        ]
        for line_index in range(max(map(len, cells))):
            lines.append(
                format_line(
                    [
                        cell[line_index] if line_index < len(cell) else ""
                        for cell in cells
                    ]
                )
            )
    return "\n".join(lines)


def export_results(rows: list[dict], output_path: str | Path) -> Path:
    """Export sorted rows with full paths, full reasons, and rounded metrics."""
    path = Path(output_path)
    if path.suffix.lower() not in {".csv", ".json"}:
        raise ValueError("output path must end in .csv or .json")
    records = []
    for row in rows:
        record = {field: row[field] for field in RESULT_FIELDS}
        record["scenario"] = _scenario_name(row["scenario"])
        for field in ("area", "perimeter", "score"):
            record[field] = round(record[field], 2)
        records.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    else:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=RESULT_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        **record,
                        "valid": str(record["valid"]).lower(),
                        **{
                            field: f"{record[field]:.2f}"
                            for field in ("area", "perimeter", "score")
                        },
                    }
                )
    return path


def best_shape_for_scenario(
    scenario_path: str | Path, max_shapes: int | None = None
) -> dict:
    if max_shapes is not None and max_shapes < 1:
        raise ValueError("max_shapes must be positive")
    best = {
        "shape": "none",
        "valid": False,
        "area": 0.0,
        "perimeter": 0.0,
        "score": 0.0,
        "reason": "",
    }
    try:
        scenario = read_scenario(str(scenario_path))
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        InvalidRoomException,
        GEOSException,
    ) as exc:
        # A bad JSON must not prevent the remaining scenarios from being evaluated.
        best["reason"] = f"scenario error: {exc}"
    else:
        player = Player10(scenario.room, scenario.inventory, scenario.weights)
        selection = player.select_best_candidate(max_shapes=max_shapes)
        if selection is not None:
            candidate, _, result, score = selection
            best = {
                "shape": candidate.name,
                "valid": True,
                "area": result.area,
                "perimeter": result.perimeter,
                "score": score,
                "reason": result.reason,
            }
        else:
            best["reason"] = player.note or "no positive-scoring candidate selected"
    return {"scenario": str(scenario_path), **best}


def _failure_category(row: dict) -> str:
    reason = str(row["reason"])
    if row["valid"]:
        return "valid"
    if reason.startswith("Not enough right connectors"):
        return "fewer than four right connectors"
    if reason.startswith(
        (
            "no rectangle:",
            "no inventory-feasible rectangle",
            "no inventory-feasible supported shape",
        )
    ):
        return "pieces cannot form a supported shape"
    if "shapes tried; none fit" in reason or "rectangles tried; none fit" in reason:
        return "no sampled placement fits the room"
    if reason.startswith(
        ("best valid rectangle", "best remaining rectangle", "best remaining shape")
    ):
        return "remaining shapes cannot beat opting out"
    if reason.startswith("scenario error:"):
        return "scenario could not be read"
    return "other no-solution result"


def run_shape_pipeline(
    scenario_path: str | Path | None = None,
    max_shapes: int | None = None,
    output_path: str | Path | None = None,
) -> list[dict]:
    if max_shapes is not None and max_shapes < 1:
        raise ValueError("max_shapes must be positive")
    paths = (
        [Path(scenario_path)] if scenario_path is not None else iter_scenario_files()
    )
    if output_path is not None:
        output = Path(output_path).resolve()
        if output.suffix.lower() not in {".csv", ".json"}:
            raise ValueError("output path must end in .csv or .json")
        if scenario_path is not None and Path(scenario_path).resolve() == output:
            raise ValueError("output path must differ from the scenario input")
        paths = [path for path in paths if path.resolve() != output]
    results = [best_shape_for_scenario(path, max_shapes) for path in paths]
    results.sort(key=lambda row: (row["valid"], row["score"]), reverse=True)
    print(_format_table(results))
    valid_count = sum(row["valid"] for row in results)
    scenario_word = "scenario" if len(results) == 1 else "scenarios"
    print(
        f"\n{len(results)} {scenario_word}: {valid_count} valid, {len(results) - valid_count} without a solution."
    )
    failures = Counter(_failure_category(row) for row in results if not row["valid"])
    if failures:
        print("No-solution breakdown:")
        for reason, count in failures.most_common():
            print(f"  {count:>2}  {reason}")
    if output_path is not None:
        exported = export_results(results, output_path)
        print(f"Exported {len(results)} rows to {exported}")
    return results


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--scenario", help="Single JSON scenario file to evaluate.")
    selection.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Evaluate every JSON under scenarios/ and players/ (the default).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Export sorted results to a .csv or .json file (creates parent directories).",
    )
    parser.add_argument(
        "--max-shapes",
        type=_positive_int,
        default=None,
        help="Cap score-first candidates per scenario; default: evaluate all candidates.",
    )
    args = parser.parse_args()
    if args.output is not None and args.output.suffix.lower() not in {".csv", ".json"}:
        parser.error("--output must end in .csv or .json")
    run_shape_pipeline(
        args.scenario, max_shapes=args.max_shapes, output_path=args.output
    )


if __name__ == "__main__":
    main()
