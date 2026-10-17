
import gymnasium as gym
from stable_baselines3 import PPO

# Show the LunarLander window
env = gym.make("LunarLander-v3", render_mode="human")

model = PPO.load(
    "ppo_lunarlander_lr_0002_entcoef01_seed42",
    env=env
)

total_episodes = 10
successful_landings = 0

for episode in range(1, total_episodes + 1):
    obs, info = env.reset()
    episode_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

    # Approximate success: LunarLander reward above 200
    success = episode_reward >= 200

    if success:
        successful_landings += 1

    print(
        f"Episode {episode}: "
        f"Reward = {episode_reward:.2f}, "
        f"Successful = {success}"
    )

print("\n--- Results ---")
print(f"Successful landings: {successful_landings}/{total_episodes}")
print(f"Success rate: {successful_landings / total_episodes * 100:.1f}%")

env.close()
