import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("LunarLander-v3")
model = PPO.load("ppo_lunarlander_lr_0002_entcoef01_seed123", env=env)

mean_reward,std_reward=evaluate_policy(
    model,
    env,
    n_eval_episodes=500,
    deterministic=True,

)

print(f"Mean reward:{mean_reward:.2f}")
print(f"Reward std :{std_reward:.2f}")

env.close()