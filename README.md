# RL Penalty Kick Simulation ⚽🥅

An AI **goalkeeper** that learns to save penalty kicks with **PPO** (Proximal Policy
Optimization) inside a **physics-based** 2D simulation.

The environment is built with [Gymnasium] + [PyMunk] for realistic ball physics
(velocity, gravity, collisions) and rendered with [PyGame]. Train the agent, evaluate
its save rate, and then challenge it yourself in a playable penalty shootout where you
click to aim your shot and the learned policy dives to stop it.

---

## Project Overview

| Component | File | Role |
|-----------|------|------|
| Physics + game | [`penalty_game.py`](penalty_game.py) | PyMunk physics world + PyGame rendering. The interactive shootout you play. |
| RL environment | [`goalkeeper_env.py`](goalkeeper_env.py) | Custom `gymnasium.Env` wrapping the physics world (state / action / reward). |
| Training | [`train.py`](train.py) | Trains a PPO agent with Stable-Baselines3 and saves the model. |
| Evaluation | [`evaluate.py`](evaluate.py) | Benchmarks the trained model over 100 episodes (save rate). |
| Deployed agent | [`goalkeeper_ai.py`](goalkeeper_ai.py) | Loads the trained policy and drives the keeper inside the live game. |

> **Note:** `goalkeeper_ai.py` defines the `GoalkeeperAgent` class used by
> `penalty_game.py`; it is not meant to be run on its own. You interact with the
> trained keeper by running `penalty_game.py`.

---

## Installation

1. **Requirements**
   - Python 3.9+ (3.10–3.13 recommended)
   - A display for the PyGame window (the game opens a real window)

2. **Create a virtual environment** (PowerShell syntax):

   ```powershell
   cd rl-penalty-kick-simulation
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**

   ```powershell
   pip install gymnasium stable-baselines3 pymunk pygame
   ```

   `stable-baselines3` pulls in PyTorch and NumPy automatically.

---

## Usage

Run the scripts from the project root (the trained model is loaded from `models/`).

### 1. Train the goalkeeper

```powershell
python train.py
```

Trains PPO for 100,000 timesteps and saves the policy to
`models/goalkeeper_ppo.zip`.

### 2. Evaluate the trained agent

```powershell
python evaluate.py
```

Plays 100 randomized episodes and prints saves, goals conceded, misses, and the
overall **save rate**.

### 3. Play against the AI keeper

```powershell
python penalty_game.py
```

A PyGame window opens.

- **Click** inside the goal to shoot toward that spot.
- The trained PPO goalkeeper reacts and dives in real time.
- Press **R** to reset.

Each result is shown on screen: **SAVED**, **GOAL**, or **MISSED**.

---

## How It Works

### Observation Space

A 5-feature `Box`, all normalized to stable ranges so the network sees
consistently scaled inputs:

| Feature | Meaning | Normalization |
|---------|---------|---------------|
| `ball.x` | Ball horizontal position | `/ WIDTH` |
| `ball.y` | Ball vertical position | `/ HEIGHT` |
| `ball.vx` | Ball horizontal velocity | `/ 1100` (shot speed) |
| `ball.vy` | Ball vertical velocity | `/ 1100` |
| `keeper.x` | Goalkeeper horizontal position | `/ WIDTH` |

### Action Space

`Discrete(3)` — a simple horizontal movement decision each step:

| Action | Behaviour |
|--------|-----------|
| `0` | Dive / move **left** |
| `1` | **Stay** (no movement) |
| `2` | Dive / move **right** |

The keeper moves at a fixed speed and is prevented from leaving the goal mouth.

### Reward Shaping

The reward encourages saves while discouraging both conceding and stalling:

| Outcome | Reward | Terminal? |
|---------|--------|-----------|
| Keeper touches the ball (**save**) | `+1.0` | Yes |
| Ball enters the goal (**conceded**) | `-1.0` | Yes |
| Ball leaves the playable area (**missed shot**) | `0.0` | Yes |
| Episode times out (240 steps) without a result | `-0.25` | Truncated |

The `-0.25` timeout penalty discourages passive policies that never commit to a dive.

### Critical Consistency Rule

The observation layout, normalization, and action→movement logic in
[`goalkeeper_ai.py`](goalkeeper_ai.py) **must stay identical** to
[`GoalkeeperEnv`](goalkeeper_env.py). If they drift, the deployed network receives
inputs it never saw during training and its decisions become meaningless.

---

## Project Structure

```
rl/
├── penalty_game.py       # Physics + PyGame interactive shootout
├── goalkeeper_env.py     # Gymnasium RL environment
├── goalkeeper_ai.py      # Deployed PPO agent (used by penalty_game.py)
├── train.py              # Train the PPO policy
├── evaluate.py           # Benchmark the trained policy
├── models/
│   └── goalkeeper_ppo.zip  # Pretrained model
└── .gitignore
```

---

## Tech Stack

`Python` · [Gymnasium] · [Stable-Baselines3 (PPO)] · [PyMunk] · [PyGame]

[//]: # (links)
[Gymnasium]: https://gymnasium.farama.org/
[Stable-Baselines3 (PPO)]: https://stable-baselines3.readthedocs.io/
[PyMunk]: https://www.pymunk.org/
[PyGame]: https://www.pygame.org/
