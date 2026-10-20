import numpy as np
from stable_baselines3 import PPO

import penalty_game as game


class GoalkeeperAgent:
    """Drive a pymunk goalkeeper body with the trained PPO policy.

    IMPORTANT: the observation layout, normalization and action->movement
    logic here MUST stay identical to ``GoalkeeperEnv`` (goalkeeper_env.py).
    If they drift, the network receives inputs it never saw during training
    and its decisions become meaningless.
    """

    # Same value GoalkeeperEnv.keeper_speed uses.
    KEEPER_SPEED = 450

    def __init__(self, model_path="models/goalkeeper_ppo"):
        self.model = PPO.load(model_path)

    def _get_obs(self, ball_body, keeper_body):
        """Build the 5-feature observation, exactly like GoalkeeperEnv._get_obs."""
        return np.array([
            ball_body.position.x / game.WIDTH,
            ball_body.position.y / game.HEIGHT,
            ball_body.velocity.x / 1100,
            ball_body.velocity.y / 1100,
            keeper_body.position.x / game.WIDTH,
        ], dtype=np.float32)

    def step(self, ball_body, keeper_body, dt):
        """Ask the policy for a move and apply it to the keeper for one frame.

        Returns the integer action taken (0 = left, 1 = stay, 2 = right).
        """
        obs = self._get_obs(ball_body, keeper_body)

        action, _ = self.model.predict(obs, deterministic=True)
        action = int(action)

        # Convert the action into a horizontal velocity (same as env.step).
        if action == 0:
            vx = -self.KEEPER_SPEED
        elif action == 2:
            vx = self.KEEPER_SPEED
        else:
            vx = 0

        # Keep the goalkeeper inside the goal (same guard as the env).
        min_x = game.GOAL_LEFT + game.KEEPER_WIDTH / 2
        max_x = game.GOAL_RIGHT - game.KEEPER_WIDTH / 2

        next_x = keeper_body.position.x + vx * dt

        if next_x < min_x or next_x > max_x:
            vx = 0

        keeper_body.velocity = (vx, 0)

        return action
