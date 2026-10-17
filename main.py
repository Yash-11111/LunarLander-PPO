# import gymnasium as gym
# from stable_baselines3 import PPO
# from stable_baselines3.common.evaluation import evaluate_policy

# env = gym.make("LunarLander-v3")

# model=PPO(
#     "MlpPolicy",
#     env,
#     learning_rate=0.0002,
#     n_steps=2048,
#     batch_size=64,
#     n_epochs=10,
#     gamma=0.99,
#     gae_lambda=0.95,
#     seed=42,
    # ent_coef=0.01,
#     verbose=1
# )

# model.learn(total_timesteps=1_000_000)

# model.save("ppo_lunarlander_lr_0002_entcoef01_seed42")


# mean_reward,std_reward=evaluate_policy(
#     model,
#     env,
#     n_eval_episodes=100,
#     deterministic=True,

# )

# print(f"Mean reward:{mean_reward:.2f}")
# print(f"Reward std :{std_reward:.2f}")

# env.close()

# above model is the best one so far






import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("LunarLander-v3")

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0002,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    seed=42,
    ent_coef=0.01,
    verbose=1
)

# Training the new model
model.learn(total_timesteps=1_000_000)

# Saving the trained model
model.save("ppo_lunarlander_lr_0002_entcoef01_seed42")

# Evaluating the same model after training
mean_reward, std_reward = evaluate_policy(
    model,
    env,
    n_eval_episodes=100,
    deterministic=True
)

print(f"Mean reward: {mean_reward:.2f}")
print(f"Reward std : {std_reward:.2f}")

env.close()