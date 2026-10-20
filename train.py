
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from goalkeeper_env import GoalkeeperEnv

import os

# Create a folder for the trained model.
os.makedirs("models", exist_ok=True)

# Create the training environment.
env = Monitor(GoalkeeperEnv())

# Create the PPO learning algorithm.
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    verbose=1,
    seed=42
)

# Train the goalkeeper.
model.learn(total_timesteps=100_000)

# Save the trained model.
model.save("models/goalkeeper_ppo")

env.close()

print("Training completed!")
print("Model saved to models/goalkeeper_ppo.zip")
