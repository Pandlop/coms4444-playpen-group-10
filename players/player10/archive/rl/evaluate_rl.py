"""Evaluate a saved Player 10 RL model with exact simulator scores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sb3_contrib import MaskablePPO

from players.player10.archive.rl.rl_env import load_training_cases
from players.player10.archive.rl.train_rl import EVALUATION_FIELDS, evaluate_policy


def _resolve_model(run_dir: Path, requested: Path | None) -> Path:
    if requested is not None:
        return requested
    for name in ("best_model.zip", "final_model.zip"):
        path = run_dir / name
        if path.exists():
            return path
    checkpoints = sorted(
        (run_dir / "checkpoints").glob("model_*_steps.zip"),
        key=lambda path: int(path.stem.split("_")[1]),
    )
    if checkpoints:
        return checkpoints[-1]
    raise FileNotFoundError(f"no model found under {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    configured_scenarios = config["scenario"]
    scenario_paths = (
        None
        if configured_scenarios == "all bundled scenarios"
        else configured_scenarios
    )
    candidate_limit = config.get("candidate_limit", config["max_candidates"])
    # Runs created before general templates used rectangles only. Preserve
    # their candidate vocabulary so saved action indices retain their meaning.
    candidate_kinds = set(config.get("candidate_kinds", ["rectangle"]))
    cases = load_training_cases(
        scenario_paths,
        max_candidates=(None if candidate_limit is None else int(candidate_limit)),
        generated_count=int(config["generated"]),
        seed=int(config["seed"]),
        progress=print,
        candidate_kinds=candidate_kinds,
    )
    env_options = {
        "max_candidates": int(config["max_candidates"]),
        "max_headings": int(config["max_headings"]),
        "max_centers": int(config["max_centers"]),
        "reward_scale": float(config["reward_scale"]),
    }
    model_path = _resolve_model(run_dir, args.model)
    model = MaskablePPO.load(model_path)
    rows, summary = evaluate_policy(
        model, cases, env_options, timesteps=int(model.num_timesteps)
    )
    rows.sort(key=lambda row: (row["valid"], row["simulator_score"]), reverse=True)
    output = (args.output or run_dir / "manual_evaluation.csv").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=EVALUATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"model: {model_path}")
    print(f"results: {output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
