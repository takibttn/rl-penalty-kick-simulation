
from stable_baselines3 import PPO
from goalkeeper_env import GoalkeeperEnv

env = GoalkeeperEnv()
model = PPO.load("models/goalkeeper_ppo")

episodes = 100

saves = 0
goals = 0
misses = 0

for episode in range(episodes):
    obs, info = env.reset(seed=1000 + episode)

    while True:
        action, _ = model.predict(
            obs,
            deterministic=True
        )
        if episode == 0:
            print(
            f"Ball=({obs[0] * 1000:.1f}, {obs[1] * 650:.1f}) | "
            f"Velocity=({obs[2] * 1100:.1f}, {obs[3] * 1100:.1f}) | "
            f"Keeper X={obs[4] * 1000:.1f} | "
            f"Action={int(action)}"
            )

        obs, reward, terminated, truncated, info = (
            env.step(int(action))
        )

        if terminated or truncated:
            if info["saved"]:
                saves += 1
            elif reward == -1.0:
                goals += 1
            else:
                misses += 1

            break

print("\nEvaluation results")
print("------------------")
print("Episodes:", episodes)
print("Saves:", saves)
print("Goals conceded:", goals)
print("Misses:", misses)
print("Save rate:", round(saves / episodes * 100, 2), "%")

env.close()
