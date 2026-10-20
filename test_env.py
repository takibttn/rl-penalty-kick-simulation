
from goalkeeper_env import GoalkeeperEnv

env = GoalkeeperEnv()

obs, info = env.reset(seed=42)

print("Initial observation:", obs)
print("Observation shape:", obs.shape)
print("Action space:", env.action_space)

for i in range(10):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = (
        env.step(action)
    )

    print(
        f"Step {i + 1}: "
        f"reward={reward}, "
        f"terminated={terminated}, "
        f"truncated={truncated}"
    )

    if terminated or truncated:
        print("Episode ended:", info)
        break

print("Environment test completed.")
