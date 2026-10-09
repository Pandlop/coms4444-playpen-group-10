# Player 10

The active strategy searches rectangles, L-shapes, U-shapes, and diagonal
octagons. It ranks candidates by simulator score, validates every placement,
and opts out when the best score is not positive.

```bash
# Run Player 10 in the UI
uv run python main.py --gui --player 10 --scenario scenarios/players/player9/crown.json

# Evaluate all scenarios and refresh the CSV
uv run python -m players.player10.tests.pipeline --all-scenarios \
  --output players/player10/tests/results/scenario_results.csv

# Run active strategy tests
uv run python -m unittest players.player10.tests.test_strategy
```

The earlier reinforcement-learning experiment is preserved in `archive/rl`.
Its large models and training logs are under `logs/archive/player10_rl`.
