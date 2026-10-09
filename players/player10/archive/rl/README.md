# Archived reinforcement-learning experiment

This masked-PPO experiment was preserved for reference. It performed worse
than the deterministic strategy and is not part of the active Player 10 path.

Source files:

- `rl_env.py`: training environment
- `train_rl.py`: trainer and progress logging
- `evaluate_rl.py`: exact simulator evaluation

Models and logs are stored in `logs/archive/player10_rl` (about 1.9 GB and
ignored by Git). The optional RL dependencies were removed from the active
project configuration. To revive this experiment, add Gymnasium,
Stable-Baselines3, SB3-Contrib, and TensorBoard to a separate environment.

Run modules through their archived paths, for example:

```bash
python -m players.player10.archive.rl.evaluate_rl \
  --run-dir logs/archive/player10_rl/bundled-v2-unlimited
```
