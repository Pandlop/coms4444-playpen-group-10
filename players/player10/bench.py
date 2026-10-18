"""Run players over scenario files and print a score table.

uv run python -m players.player10.bench                 # player 10 vs 4, all scenarios
uv run python -m players.player10.bench -p 10 4 5 -j 4 scenarios/players/player6/*.json
"""

import argparse
import glob
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCORE = re.compile(r"SCORE:\s*(-?[\d.]+)")
CPU = re.compile(r"player CPU time:\s*([\d.]+)s")


def run(player: str, scenario: str, log_dir: str) -> tuple[float | None, float | None]:
    out = subprocess.run(
        [sys.executable, "main.py", "-p", player, "-i", scenario, "--log-dir", log_dir],
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    score, cpu = SCORE.search(out), CPU.search(out)
    return (
        float(score.group(1)) if score else None,
        float(cpu.group(1)) if cpu else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenarios", nargs="*")
    parser.add_argument("-p", "--players", nargs="+", default=["10", "4"])
    parser.add_argument("-j", "--jobs", type=int, default=4)
    args = parser.parse_args()
    scenarios = args.scenarios or sorted(
        p
        for p in glob.glob("scenarios/**/*.json", recursive=True)
        if "invalid" not in p
    )
    log_dir = tempfile.mkdtemp(prefix="bench-logs-")
    jobs = [(p, s) for s in scenarios for p in args.players]
    with ThreadPoolExecutor(args.jobs) as pool:
        results = dict(zip(jobs, pool.map(lambda job: run(*job, log_dir), jobs)))

    name_width = max(len(Path(s).with_suffix("").as_posix()) for s in scenarios) - 10
    header = f"{'scenario':<{name_width}}" + "".join(
        f"{'p' + p:>10}{'cpu':>7}" for p in args.players
    )
    print(header)
    print("-" * len(header))
    wins = dict.fromkeys(args.players, 0)
    totals = dict.fromkeys(args.players, 0.0)
    for s in scenarios:
        row = f"{Path(s).with_suffix('').as_posix().removeprefix('scenarios/'):<{name_width}}"
        scores = {p: results[(p, s)][0] for p in args.players}
        best = max((v for v in scores.values() if v is not None), default=None)
        for p in args.players:
            score, cpu = results[(p, s)]
            mark = "*" if score is not None and score == best and best > 0 else " "
            row += f"{'-' if score is None else f'{score:.1f}':>9}{mark}"
            row += f"{'-' if cpu is None else f'{cpu:.0f}s':>7}"
            totals[p] += score or 0.0
            wins[p] += mark == "*"
        print(row)
    print("-" * len(header))
    print(
        "total / best-or-tied: "
        + ", ".join(f"p{p} {totals[p]:.0f} / {wins[p]}" for p in args.players)
    )


if __name__ == "__main__":
    main()
