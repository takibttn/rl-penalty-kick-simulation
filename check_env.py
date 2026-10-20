
from stable_baselines3.common.env_checker import check_env
from goalkeeper_env import GoalkeeperEnv

env = GoalkeeperEnv()

check_env(env)

print("Environment passed the compatibility checks!")


obs, info = env.reset(seed=42)

for step in range(240):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    if not env.observation_space.contains(obs):
        print("Invalid observation:", obs)
        break

    if terminated or truncated:
        obs, info = env.reset()
else:
    print("All observations stayed within the declared bounds.")

env.close()
