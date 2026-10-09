"""Train and evaluate a masked PPO policy for Player 10 rectangles."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from players.player10.archive.rl.rl_env import (
    PlaypenRectangleEnv,
    TrainingCase,
    load_training_cases,
)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RUNS_DIR = ROOT / "logs" / "archive" / "player10_rl"
PROGRESS_FIELDS = [
    "elapsed_seconds",
    "timesteps",
    "episodes",
    "window_episodes",
    "mean_training_reward",
    "mean_simulator_score",
    "best_simulator_score",
    "valid_rate",
    "opt_out_rate",
]
EVALUATION_FIELDS = [
    "timesteps",
    "scenario",
    "shape",
    "valid",
    "opted_out",
    "area",
    "perimeter",
    "simulator_score",
    "heading",
    "center_x",
    "center_y",
    "reason",
]


def _append_csv(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_policy_episode(
    model: MaskablePPO,
    env: PlaypenRectangleEnv,
    case_index: int,
) -> dict[str, Any]:
    observation, _ = env.reset(options={"case_index": case_index})
    terminated = truncated = False
    info: dict[str, Any] = {}
    while not (terminated or truncated):
        action, _ = model.predict(
            observation,
            action_masks=env.action_masks(),
            deterministic=True,
        )
        observation, _, terminated, truncated, info = env.step(
            int(np.asarray(action).item())
        )
    return info


def evaluate_policy(
    model: MaskablePPO,
    cases: list[TrainingCase],
    env_options: dict[str, Any],
    *,
    timesteps: int,
    output_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    env = PlaypenRectangleEnv(cases, **env_options)
    rows = []
    for case_index in range(len(cases)):
        info = run_policy_episode(model, env, case_index)
        center = info.get("center")
        row = {
            "timesteps": timesteps,
            "scenario": info["scenario"],
            "shape": info["shape"],
            "valid": bool(info["valid"]),
            "opted_out": bool(info["opted_out"]),
            "area": round(float(info["area"]), 2),
            "perimeter": round(float(info["perimeter"]), 2),
            "simulator_score": round(float(info["simulator_score"]), 2),
            "heading": info.get("heading"),
            "center_x": None if center is None else round(float(center[0]), 4),
            "center_y": None if center is None else round(float(center[1]), 4),
            "reason": info["reason"],
        }
        rows.append(row)
        if output_path is not None:
            _append_csv(output_path, EVALUATION_FIELDS, row)
    scores = [row["simulator_score"] for row in rows]
    summary = {
        "mean_score": float(np.mean(scores)),
        "best_score": float(np.max(scores)),
        "valid_rate": sum(row["valid"] for row in rows) / len(rows),
        "opt_out_rate": sum(row["opted_out"] for row in rows) / len(rows),
    }
    env.close()
    return rows, summary


class ProgressCallback(BaseCallback):
    """Durable episode, progress, checkpoint, and evaluation logging."""

    def __init__(
        self,
        run_dir: Path,
        cases: list[TrainingCase],
        env_options: dict[str, Any],
        *,
        log_every_episodes: int,
        checkpoint_every: int,
        eval_every: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.run_dir = run_dir
        self.cases = cases
        self.env_options = env_options
        self.log_every_episodes = log_every_episodes
        self.checkpoint_every = checkpoint_every
        self.eval_every = eval_every
        self.started_at = time.monotonic()
        self.episodes = 0
        self.window: deque[dict[str, Any]] = deque(maxlen=200)
        self.next_checkpoint = checkpoint_every
        self.next_eval = eval_every
        self.best_eval_mean = float("-inf")
        self._episode_stream = (run_dir / "episodes.jsonl").open("a", encoding="utf-8")

    def _on_training_start(self) -> None:
        current = int(self.model.num_timesteps)
        self.next_checkpoint = (
            (current // self.checkpoint_every) + 1
        ) * self.checkpoint_every
        self.next_eval = ((current // self.eval_every) + 1) * self.eval_every

    def _write_progress(self) -> None:
        records = list(self.window)
        if not records:
            return
        row = {
            "elapsed_seconds": round(time.monotonic() - self.started_at, 2),
            "timesteps": int(self.num_timesteps),
            "episodes": self.episodes,
            "window_episodes": len(records),
            "mean_training_reward": round(
                float(np.mean([item["training_reward"] for item in records])), 4
            ),
            "mean_simulator_score": round(
                float(np.mean([item["simulator_score"] for item in records])), 2
            ),
            "best_simulator_score": round(
                float(max(item["simulator_score"] for item in records)), 2
            ),
            "valid_rate": round(
                sum(item["valid"] for item in records) / len(records), 4
            ),
            "opt_out_rate": round(
                sum(item["opted_out"] for item in records) / len(records), 4
            ),
        }
        _append_csv(self.run_dir / "progress.csv", PROGRESS_FIELDS, row)
        print(
            "training: "
            f"steps={row['timesteps']} episodes={row['episodes']} "
            f"mean_score={row['mean_simulator_score']:.2f} "
            f"valid={row['valid_rate']:.1%} opt_out={row['opt_out_rate']:.1%}"
        )

    def _run_evaluation(self) -> None:
        _, summary = evaluate_policy(
            self.model,
            self.cases,
            self.env_options,
            timesteps=int(self.num_timesteps),
            output_path=self.run_dir / "evaluation_details.csv",
        )
        row = {"timesteps": int(self.num_timesteps), **summary}
        _append_csv(
            self.run_dir / "evaluations.csv",
            ["timesteps", "mean_score", "best_score", "valid_rate", "opt_out_rate"],
            row,
        )
        print(
            "evaluation: "
            f"steps={self.num_timesteps} mean_score={summary['mean_score']:.2f} "
            f"valid={summary['valid_rate']:.1%} opt_out={summary['opt_out_rate']:.1%}"
        )
        if summary["mean_score"] > self.best_eval_mean:
            self.best_eval_mean = summary["mean_score"]
            self.model.save(self.run_dir / "best_model")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        for index, info in enumerate(infos):
            if not info.get("terminal"):
                continue
            self.episodes += 1
            record = {
                "episode": self.episodes,
                "timesteps": int(self.num_timesteps),
                "elapsed_seconds": round(time.monotonic() - self.started_at, 3),
                "training_reward": float(rewards[index]),
                **{
                    key: value
                    for key, value in info.items()
                    if key
                    in {
                        "scenario",
                        "shape",
                        "candidate_index",
                        "heading",
                        "center",
                        "valid",
                        "opted_out",
                        "simulator_score",
                        "area",
                        "perimeter",
                        "inside_fraction",
                        "reason",
                        "masked_action",
                    }
                },
            }
            self.window.append(record)
            self._episode_stream.write(json.dumps(record, sort_keys=True) + "\n")
            self._episode_stream.flush()
            if self.episodes % self.log_every_episodes == 0:
                self._write_progress()

        if self.num_timesteps >= self.next_checkpoint:
            self.model.save(
                self.run_dir / "checkpoints" / f"model_{self.num_timesteps}_steps"
            )
            self.next_checkpoint += self.checkpoint_every
        if self.num_timesteps >= self.next_eval:
            self._run_evaluation()
            self.next_eval += self.eval_every
        return True

    def close(self) -> None:
        if not self._episode_stream.closed:
            self._episode_stream.close()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _default_run_dir() -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return DEFAULT_RUNS_DIR / stamp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        action="append",
        help="Train on this scenario; repeat for multiple paths. Default: all bundled.",
    )
    parser.add_argument("--generated", type=_nonnegative_int, default=0)
    parser.add_argument("--timesteps", type=_positive_int, default=100_000)
    parser.add_argument(
        "--max-candidates",
        type=_positive_int,
        default=None,
        help="Optional experimental cap; default: expose every candidate.",
    )
    parser.add_argument(
        "--max-headings",
        type=_positive_int,
        default=None,
        help="Optional experimental cap; default: infer enough for every scenario.",
    )
    parser.add_argument(
        "--max-centers",
        type=_positive_int,
        default=None,
        help="Optional experimental cap; default: infer enough for every placement.",
    )
    parser.add_argument("--reward-scale", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--n-steps", type=_positive_int, default=1024)
    parser.add_argument("--batch-size", type=_positive_int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--log-every", type=_positive_int, default=100)
    parser.add_argument("--checkpoint-every", type=_positive_int, default=10_000)
    parser.add_argument("--eval-every", type=_positive_int, default=10_000)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--check-env", action="store_true", help="Run Gymnasium compatibility checks."
    )
    args = parser.parse_args()

    if args.reward_scale <= 0:
        parser.error("--reward-scale must be positive")
    if args.batch_size > args.n_steps:
        parser.error("--batch-size cannot exceed --n-steps")
    run_dir = (args.run_dir or _default_run_dir()).resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and args.resume is None:
        parser.error(f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    print(f"run directory: {run_dir}")

    # A resumed policy must keep the original observation and action sizes.
    previous_config_path = run_dir / "run_config.json"
    if args.resume is not None and previous_config_path.exists():
        previous = json.loads(previous_config_path.read_text(encoding="utf-8"))
        if args.max_candidates is None:
            args.max_candidates = previous.get(
                "candidate_limit", previous["max_candidates"]
            )
        if args.max_headings is None:
            args.max_headings = int(previous["max_headings"])
        if args.max_centers is None:
            args.max_centers = int(previous["max_centers"])

    cases = load_training_cases(
        args.scenario,
        max_candidates=args.max_candidates,
        generated_count=args.generated,
        seed=args.seed,
        progress=print,
    )
    candidate_counts = [len(case.candidates) for case in cases]
    print(
        f"prepared {len(cases)} scenarios; candidate counts "
        f"min={min(candidate_counts)}, max={max(candidate_counts)}, "
        f"mean={np.mean(candidate_counts):.1f}"
    )
    requested_candidate_limit = args.max_candidates
    layout_env = PlaypenRectangleEnv(
        cases,
        max_candidates=args.max_candidates,
        max_headings=args.max_headings,
        max_centers=args.max_centers,
        reward_scale=args.reward_scale,
    )
    env_options = {
        "max_candidates": layout_env.max_candidates,
        "max_headings": layout_env.max_headings,
        "max_centers": layout_env.max_centers,
        "reward_scale": args.reward_scale,
    }
    layout_env.close()
    print(
        "action capacities: "
        f"candidates={env_options['max_candidates']}, "
        f"headings={env_options['max_headings']}, "
        f"centers={env_options['max_centers']}"
    )
    config = {
        **vars(args),
        "candidate_limit": requested_candidate_limit,
        "max_candidates": env_options["max_candidates"],
        "max_headings": env_options["max_headings"],
        "max_centers": env_options["max_centers"],
        "run_dir": str(run_dir),
        "resume": None if args.resume is None else str(args.resume.resolve()),
        "scenario": args.scenario or "all bundled scenarios",
        "case_names": [case.name for case in cases],
        "candidate_kinds": sorted(
            {candidate.kind for case in cases for candidate in case.candidates}
        ),
        "observation": "room, weights, inventory, candidates, headings, centers",
        "actions": "candidate or opt out, heading, center",
        "terminal_score": "src.enclosure.score_construction",
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8"
    )

    if args.check_env:
        check_env(PlaypenRectangleEnv(cases, **env_options), warn=True)

    def make_env():
        return RecordEpisodeStatistics(PlaypenRectangleEnv(cases, **env_options))

    vector_env = DummyVecEnv([make_env])
    if args.resume is None:
        model = MaskablePPO(
            "MlpPolicy",
            vector_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            seed=args.seed,
            verbose=1,
            tensorboard_log=str(run_dir / "tensorboard"),
            policy_kwargs={"net_arch": [256, 256]},
        )
    else:
        model = MaskablePPO.load(
            args.resume,
            env=vector_env,
            tensorboard_log=str(run_dir / "tensorboard"),
        )
    model.set_logger(configure(str(run_dir / "sb3"), ["stdout", "csv", "tensorboard"]))

    callback = ProgressCallback(
        run_dir,
        cases,
        env_options,
        log_every_episodes=args.log_every,
        checkpoint_every=args.checkpoint_every,
        eval_every=args.eval_every,
    )
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            reset_num_timesteps=args.resume is None,
            tb_log_name="masked_ppo",
        )
        model.save(run_dir / "final_model")
        rows, summary = evaluate_policy(
            model,
            cases,
            env_options,
            timesteps=int(model.num_timesteps),
        )
        if summary["mean_score"] > callback.best_eval_mean:
            callback.best_eval_mean = summary["mean_score"]
            model.save(run_dir / "best_model")
        with (run_dir / "final_evaluation.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=EVALUATION_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        (run_dir / "final_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(f"finished: model and logs written to {run_dir}")
        print(json.dumps(summary, indent=2))
    finally:
        callback.close()
        vector_env.close()


if __name__ == "__main__":
    main()
