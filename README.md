# LunarLander-PPO

Training and evaluating a Proximal Policy Optimization (PPO) agent on Gymnasium's `LunarLander-v3` using Stable-Baselines3.

The agent controls a lunar lander using discrete actions (fire the main engine, fire the left or right engine, or do nothing). It is rewarded for landing softly between the flags and penalised for crashing or wasting fuel. An episode reward of 200 or more is generally treated as a successful landing.

## Project structure

| File | Purpose |
| --- | --- |
| `main.py` | Trains the PPO agent, saves the model, then evaluates it |
| `evaluate.py` | Loads a saved model and evaluates it over many episodes |
| `play.py` | Renders the trained agent landing and reports its success rate |
| `requirements.txt` | Python dependencies |

## Setup

```bash
git clone https://github.com/Yash-11111/LunarLander-PPO.git
cd LunarLander-PPO
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`LunarLander-v3` needs Box2D. If the environment fails to build, install it with:

```bash
pip install "gymnasium[box2d]"
```

## Usage

### 1. Train

```bash
python main.py
```

This trains for 1,000,000 timesteps, saves the model as `ppo_lunarlander_lr_0002_entcoef01_seed42.zip`, and then evaluates it over 100 deterministic episodes, printing the mean reward and standard deviation.

### 2. Evaluate

```bash
python evaluate.py
```

Evaluates a saved model over 500 deterministic episodes. Make sure the model name inside `evaluate.py` matches the file you saved in step 1.

### 3. Watch the agent play

```bash
python play.py
```

Opens the LunarLander window and runs 10 episodes. An episode counts as a successful landing when its reward is 200 or more, and the script prints the overall success rate at the end.

## Hyperparameters

| Parameter | Value |
| --- | --- |
| Policy | `MlpPolicy` |
| Learning rate | 0.0002 |
| `n_steps` | 2048 |
| Batch size | 64 |
| Epochs per update | 10 |
| Discount factor (`gamma`) | 0.99 |
| GAE lambda | 0.95 |
| Entropy coefficient | 0.01 |
| Seed | 42 |
| Total timesteps | 1,000,000 |

## Results

Best run, evaluated over 500 deterministic episodes with `evaluate.py`:

| Metric | Value |
| --- | --- |
| Mean reward | 277.66 |
| Reward std | 35.92 |
| Learning rate | 0.0002 |
| Entropy coefficient | 0.01 |
| Seed | 42 |
| Training timesteps | 1,000,000 |

A mean reward above 200 is generally considered a solved landing, and the low standard deviation shows the agent lands consistently rather than occasionally.

## Tech stack

- Python
- Gymnasium (`LunarLander-v3`)
- Stable-Baselines3 (PPO)
