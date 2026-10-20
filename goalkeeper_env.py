import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pymunk
import math

import penalty_game as game

class GoalkeeperEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=np.array([0, 0, -2, -2, 0], dtype=np.float32),
            high=np.array([1, 1, 2, 2, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.dt = 1 / 60
        self.keeper_speed = 450
        self.max_steps = 240
        self.steps = 0

        self.space = None
        self.ball_body = None
        self.keeper_body = None
        self.keeper_shape = None
        self.keeper_touched = False

    def _get_obs(self):
        ball = self.ball_body
        keeper = self.keeper_body

        return np.array([
            ball.position.x / game.WIDTH,
            ball.position.y / game.HEIGHT,
            ball.velocity.x / 1100,
            ball.velocity.y / 1100,
            keeper.position.x / game.WIDTH
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Build a fresh physics world for each penalty.
        self.space = pymunk.Space()
        self.space.gravity = (0, game.GRAVITY)

        game.create_goal(self.space)

        self.keeper_body, self.keeper_shape = (
            game.create_goalkeeper(self.space)
        )

        self.ball_body, _ = game.create_ball(self.space)

        self.keeper_body.position = (
            game.WIDTH // 2,
            game.GOAL_BOTTOM - game.KEEPER_HEIGHT // 2
        )

        self.keeper_body.velocity = (0, 0)
        self.keeper_touched = False
        self.steps = 0

        # Randomize the shot for every episode.
        target_x = self.np_random.uniform(
            game.GOAL_LEFT + 25,
            game.GOAL_RIGHT - 25
        )

        target_y = self.np_random.uniform(
            game.GOAL_TOP + 20,
            game.GOAL_BOTTOM - 70
        )

        game.shoot_ball(
            self.ball_body,
            target_x,
            target_y
        )

        return self._get_obs(), {}
        
    def step(self, action):
        self.steps += 1

        # Convert the agent's action into keeper movement.
        if action == 0:
            vx = -self.keeper_speed
        elif action == 2:
            vx = self.keeper_speed
        else:
            vx = 0

        self.keeper_body.velocity = (vx, 0)

        # Keep the goalkeeper inside the goal.
        min_x = game.GOAL_LEFT + game.KEEPER_WIDTH / 2
        max_x = game.GOAL_RIGHT - game.KEEPER_WIDTH / 2

        next_x = (
            self.keeper_body.position.x
            + vx * self.dt
        )

        if next_x < min_x or next_x > max_x:
            self.keeper_body.velocity = (0, 0)

        self.space.step(self.dt)

        bx = self.ball_body.position.x
        by = self.ball_body.position.y
        kx, ky = self.keeper_body.position

        # Detect contact between the ball and goalkeeper.
        left = kx - game.KEEPER_WIDTH / 2
        right = kx + game.KEEPER_WIDTH / 2
        top = ky - game.KEEPER_HEIGHT / 2
        bottom = ky + game.KEEPER_HEIGHT / 2

        closest_x = min(max(bx, left), right)
        closest_y = min(max(by, top), bottom)

        dx = bx - closest_x
        dy = by - closest_y

        if (
            dx * dx + dy * dy
            <= game.BALL_RADIUS ** 2
        ):
            self.keeper_touched = True

        reward = 0.0
        terminated = False

        # A save is a successful terminal state.
        if self.keeper_touched:
            reward = 1.0
            terminated = True

        # The ball entered the goal without being saved.
        elif (
            game.GOAL_LEFT < bx < game.GOAL_RIGHT
            and game.GOAL_TOP < by < game.GOAL_BOTTOM
        ):
            reward = -1.0
            terminated = True

        # The ball left the playable area.
        elif (
            bx < -game.BALL_RADIUS
            or bx > game.WIDTH + game.BALL_RADIUS
            or by < -game.BALL_RADIUS
            or by > game.HEIGHT + game.BALL_RADIUS
        ):
            reward = 0
            terminated = True

        # Avoid episodes that continue indefinitely.
        truncated = self.steps >= self.max_steps

        if truncated and not terminated:
            reward = -0.25

        info = {
            "saved": self.keeper_touched,
            "keeper_x": float(kx),
            "ball_x": float(bx),
            "ball_y": float(by)
        }

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            info
        )
