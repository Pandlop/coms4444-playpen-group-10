"""Run Player10 on every scenario and write one CSV of its results.

    uv run python players/player10/run_all.py            # 3 rooms at a time
    uv run python players/player10/run_all.py --jobs 2 --out results.csv
    uv run python players/player10/run_all.py players/player6/hallway players/player2/cool

Each room runs in its own process, so its CPU time is its own.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ["scenario", "score", "valid", "cpu_seconds", "wall_seconds", "area", "perimeter",
          "pieces", "gates", "second_gate_bonus", "method"]

ONE = r"""
import json, sys, time
sys.path.insert(0, sys.argv[2])
from src.scenario import read_scenario
from src.enclosure import validate_construction, score_construction
from players.player10.player import Player10
sc = read_scenario(sys.argv[1])
t, w = time.process_time(), time.time()
p = Player10(sc.room, sc.inventory, sc.weights)
b = p.build_enclosure()
out = {"cpu_seconds": round(time.process_time() - t, 1), "wall_seconds": round(time.time() - w, 1), "method": p.note or ""}
if b is None:
    out.update(score=0.0, valid=False, area="", perimeter="", pieces=0, gates="", second_gate_bonus="")
else:
    r = validate_construction(b, sc.room, sc.inventory)
    out.update(score=round(score_construction(r, sc.weights), 1), valid=r.valid, area=round(r.area, 1),
               perimeter=round(r.perimeter, 1), pieces=len(b.pieces), gates=r.gate_count,
               second_gate_bonus=r.second_gate_bonus)
print("RESULT" + json.dumps(out))
"""


def run(path: Path) -> dict:
    name = str(path.relative_to(ROOT / "scenarios")).removesuffix(".json")
    proc = subprocess.run([sys.executable, "-c", ONE, str(path), str(ROOT)], cwd=ROOT,
                          capture_output=True, text=True)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT")), None)
    row = {"scenario": name}
    if line is None:
        row.update(score=0.0, valid=False, method="crashed: " + proc.stderr.strip()[-200:])
    else:
        row.update(json.loads(line[len("RESULT"):]))
    print(f"{name:55s} {row['score']:>12} {row.get('cpu_seconds', '')}s", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "player10_results.csv"))
    ap.add_argument("scenarios", nargs="*", help="only these (e.g. players/player6/hallway); default all")
    args = ap.parse_args()
    paths = sorted((ROOT / "scenarios").rglob("*.json"))
    if args.scenarios:
        wanted = {s.removesuffix(".json").removeprefix("scenarios/") for s in args.scenarios}
        paths = [p for p in paths if str(p.relative_to(ROOT / "scenarios")).removesuffix(".json") in wanted]
    start = time.time()
    with ThreadPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(run, paths))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    total = sum(float(r["score"]) for r in rows)
    print(f"\n{len(rows)} rooms, total score {total:,.1f}, {time.time() - start:.0f}s wall -> {args.out}")


if __name__ == "__main__":
    main()
