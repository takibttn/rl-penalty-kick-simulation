import pygame
import pymunk
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

WIDTH = 1000
HEIGHT = 650
FPS = 120

# Goal dimensions
GOAL_WIDTH = 360
GOAL_HEIGHT = 180

GOAL_LEFT = (WIDTH - GOAL_WIDTH) // 2
GOAL_RIGHT = GOAL_LEFT + GOAL_WIDTH
GOAL_TOP = 80
GOAL_BOTTOM = GOAL_TOP + GOAL_HEIGHT

# Ball
BALL_RADIUS = 12
BALL_MASS = 0.45

# Goalkeeper
KEEPER_WIDTH = 35
KEEPER_HEIGHT = 80

# Physics
GRAVITY = 900


# ============================================================
# COLORS
# ============================================================

GREEN = (40, 140, 70)
WHITE = (240, 240, 240)
BLACK = (20, 20, 20)
RED = (210, 50, 50)
BLUE = (50, 100, 220)
YELLOW = (240, 210, 50)


# ============================================================
# BALL
# ============================================================

def create_ball(space):

    body = pymunk.Body(
        BALL_MASS,
        pymunk.moment_for_circle(
            BALL_MASS,
            0,
            BALL_RADIUS
        )
    )

    # Start near the penalty spot
    body.position = (WIDTH // 2, HEIGHT - 100)

    # Keep ball steady (no gravity) until the player shoots
    # Stay DYNAMIC but ignore gravity/damping until shot
    def _hold(body, gravity, damping, dt):
        return
    body.velocity_func = _hold
    body.velocity = (0, 0)
    body.angular_velocity = 0

    shape = pymunk.Circle(
        body,
        BALL_RADIUS
    )

    shape.elasticity = 0.5
    shape.friction = 0.5
    shape.collision_type = 1

    space.add(body, shape)

    return body, shape


# ============================================================
# GOALKEEPER
# ============================================================

def create_goalkeeper(space):

    body = pymunk.Body(
        5,
        pymunk.moment_for_box(
            5,
            (KEEPER_WIDTH, KEEPER_HEIGHT)
        ),
        body_type=pymunk.Body.KINEMATIC
    )

    body.position = (
        WIDTH // 2,
        GOAL_BOTTOM - KEEPER_HEIGHT // 2
    )

    shape = pymunk.Poly.create_box(
        body,
        (KEEPER_WIDTH, KEEPER_HEIGHT)
    )

    shape.elasticity = 0.2
    shape.friction = 0.5
    shape.collision_type = 2

    space.add(body, shape)

    return body, shape


# ============================================================
# GOAL POSTS
# ============================================================

def create_goal(space):

    # Left post
    left_body = pymunk.Body(body_type=pymunk.Body.STATIC)

    left_shape = pymunk.Segment(
        left_body,
        (GOAL_LEFT, GOAL_TOP),
        (GOAL_LEFT, GOAL_BOTTOM),
        8
    )

    # Right post
    right_body = pymunk.Body(body_type=pymunk.Body.STATIC)

    right_shape = pymunk.Segment(
        right_body,
        (GOAL_RIGHT, GOAL_TOP),
        (GOAL_RIGHT, GOAL_BOTTOM),
        8
    )

    # Crossbar
    crossbar_body = pymunk.Body(body_type=pymunk.Body.STATIC)

    crossbar_shape = pymunk.Segment(
        crossbar_body,
        (GOAL_LEFT, GOAL_TOP),
        (GOAL_RIGHT, GOAL_TOP),
        8
    )

    for shape in [
        left_shape,
        right_shape,
        crossbar_shape
    ]:
        shape.elasticity = 0.7
        shape.friction = 0.5

    space.add(
        left_body,
        left_shape,
        right_body,
        right_shape,
        crossbar_body,
        crossbar_shape
    )


# ============================================================
# DRAWING
# ============================================================

def draw_goal(screen):

    # Goal net
    pygame.draw.rect(
        screen,
        WHITE,
        (
            GOAL_LEFT,
            GOAL_TOP,
            GOAL_WIDTH,
            GOAL_HEIGHT
        ),
        3
    )

    # Net lines
    for x in range(GOAL_LEFT, GOAL_RIGHT + 1, 30):

        pygame.draw.line(
            screen,
            (150, 150, 150),
            (x, GOAL_TOP),
            (x, GOAL_BOTTOM),
            1
        )

    for y in range(GOAL_TOP, GOAL_BOTTOM + 1, 30):

        pygame.draw.line(
            screen,
            (150, 150, 150),
            (GOAL_LEFT, y),
            (GOAL_RIGHT, y),
            1
        )


def draw_ball(screen, ball_body):

    # Guard against NaN (can happen if body goes invalid)
    if math.isnan(ball_body.position.x) or math.isnan(ball_body.position.y):
        return
    x = int(ball_body.position.x)
    y = int(ball_body.position.y)

    pygame.draw.circle(
        screen,
        WHITE,
        (x, y),
        BALL_RADIUS
    )

    pygame.draw.circle(
        screen,
        BLACK,
        (x, y),
        BALL_RADIUS,
        2
    )


def draw_goalkeeper(screen, goalkeeper_body):

    x = int(goalkeeper_body.position.x)
    y = int(goalkeeper_body.position.y)

    rect = pygame.Rect(
        x - KEEPER_WIDTH // 2,
        y - KEEPER_HEIGHT // 2,
        KEEPER_WIDTH,
        KEEPER_HEIGHT
    )

    pygame.draw.rect(
        screen,
        BLUE,
        rect
    )


def draw_penalty_spot(screen):

    pygame.draw.circle(
        screen,
        WHITE,
        (WIDTH // 2, HEIGHT - 100),
        5
    )


# ============================================================
# SHOOT THE BALL
# ============================================================

def shoot_ball(ball_body, target_x, target_y):

    ball_x = ball_body.position.x
    ball_y = ball_body.position.y

    dx = target_x - ball_x
    dy = target_y - ball_y

    distance = math.sqrt(dx * dx + dy * dy)

    if distance == 0:
        return

    # Normalize direction
    dx /= distance
    dy /= distance

    # Shot power
    speed = 1100

    # Enable physics (gravity, collisions) only when shot
    ball_body.velocity_func = pymunk.Body.update_velocity
    ball_body.angular_velocity = 0
    ball_body.velocity = (
        dx * speed,
        dy * speed
    )


# ============================================================
# MAIN
# ============================================================

def main():

    pygame.init()

    screen = pygame.display.set_mode(
        (WIDTH, HEIGHT)
    )

    pygame.display.set_caption(
        "Penalty Kick Simulation"
    )

    clock = pygame.time.Clock()

    # --------------------------------------------------------
    # Physics world
    # --------------------------------------------------------

    space = pymunk.Space()

    # Gravity
    space.gravity = (0, GRAVITY)

    # Create objects
    create_goal(space)

    goalkeeper_body, goalkeeper_shape = \
        create_goalkeeper(space)

    ball_body, ball_shape = \
        create_ball(space)

    # --------------------------------------------------------
    # AI goalkeeper (trained PPO policy)
    # --------------------------------------------------------
    # Imported lazily so penalty_game and goalkeeper_ai do not
    # form a circular import at module load time.
    try:
        from goalkeeper_ai import GoalkeeperAgent
        keeper_agent = GoalkeeperAgent()
        ai_enabled = True
    except Exception as exc:
        print("AI goalkeeper disabled:", exc)
        keeper_agent = None
        ai_enabled = False

    # Flag latched when ball touches keeper (manual check for pymunk 7)
    keeper_touched = {"value": False}

    def reset_ball():
        def _hold(body, gravity, damping, dt):
            return
        ball_body.velocity_func = _hold
        ball_body.position = (WIDTH // 2, HEIGHT - 100)
        ball_body.velocity = (0, 0)
        ball_body.angular_velocity = 0
        ball_body.force = (0, 0)
        goalkeeper_body.position = (
            WIDTH // 2,
            GOAL_BOTTOM - KEEPER_HEIGHT // 2
        )
        goalkeeper_body.velocity = (0, 0)
        keeper_touched["value"] = False

    # --------------------------------------------------------
    # Game state
    # --------------------------------------------------------

    shot_taken = False
    result = None
    result_start = 0
    ball_visible = True
    shot_time = 0

    running = True

    while running:

        # ====================================================
        # EVENTS
        # ====================================================

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            # Shoot with mouse
            if event.type == pygame.MOUSEBUTTONDOWN:

                if not shot_taken and result is None and ball_visible:

                    target_x, target_y = pygame.mouse.get_pos()

                    shoot_ball(
                        ball_body,
                        target_x,
                        target_y
                    )

                    shot_taken = True
                    shot_time = pygame.time.get_ticks()
                    keeper_touched["value"] = False

            # Reset
            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_r:

                    reset_ball()
                    shot_taken = False
                    result = None
                    ball_visible = True

        # ====================================================
        # PHYSICS
        # ====================================================

        dt = 1 / FPS

        # ------------------------------------------------
        # Let the trained policy move the keeper each frame
        # while a shot is live. Before the shot, it holds
        # the centre line.
        # ------------------------------------------------
        if ai_enabled and shot_taken and result is None:
            keeper_agent.step(
                ball_body,
                goalkeeper_body,
                dt
            )
        else:
            goalkeeper_body.velocity = (0, 0)

        space.step(dt)

        # ====================================================
        # RESULT DETECTION (only after a shot)
        # ====================================================

        if shot_taken and result is None:
            bx = ball_body.position.x
            by = ball_body.position.y

            # Manual keeper collision check (pymunk 7 has no add_collision_handler)
            if not keeper_touched["value"] and not math.isnan(bx) and not math.isnan(by):
                kx, ky = goalkeeper_body.position
                left = kx - KEEPER_WIDTH / 2
                right = kx + KEEPER_WIDTH / 2
                top = ky - KEEPER_HEIGHT / 2
                bottom = ky + KEEPER_HEIGHT / 2
                closest_x = left if bx < left else (right if bx > right else bx)
                closest_y = top if by < top else (bottom if by > bottom else by)
                dx = bx - closest_x
                dy = by - closest_y
                if dx * dx + dy * dy <= BALL_RADIUS * BALL_RADIUS:
                    keeper_touched["value"] = True

            # NaN guard
            if math.isnan(bx) or math.isnan(by):
                result = "MISSED"
                result_start = pygame.time.get_ticks()
                ball_visible = False
            elif keeper_touched["value"]:
                result = "SAVED"
                result_start = pygame.time.get_ticks()
                ball_visible = False
            elif GOAL_LEFT < bx < GOAL_RIGHT and GOAL_TOP < by < GOAL_BOTTOM:
                result = "GOAL"
                result_start = pygame.time.get_ticks()
                ball_visible = False
            elif bx < -BALL_RADIUS or bx > WIDTH + BALL_RADIUS or by < -BALL_RADIUS or by > HEIGHT + BALL_RADIUS:
                result = "MISSED"
                result_start = pygame.time.get_ticks()
                ball_visible = False
            elif pygame.time.get_ticks() - shot_time > 2500:
                # Ball stopped / fell without entering goal
                vel = math.hypot(ball_body.velocity.x, ball_body.velocity.y)
                if vel < 20 or by > HEIGHT - 30:
                    result = "MISSED"
                    result_start = pygame.time.get_ticks()
                    ball_visible = False

        # Auto reset after showing result
        if result is not None:
            if pygame.time.get_ticks() - result_start > 1500:
                reset_ball()
                shot_taken = False
                result = None
                ball_visible = True

        # ====================================================
        # DRAW
        # ====================================================

        screen.fill(GREEN)

        draw_goal(screen)

        draw_penalty_spot(screen)

        draw_goalkeeper(
            screen,
            goalkeeper_body
        )

        if ball_visible:
            draw_ball(
                screen,
                ball_body
            )

        # Instructions
        font = pygame.font.Font(None, 32)

        mode = "AI goalkeeper" if ai_enabled else "manual keeper"
        text = font.render(
            f"Click inside the goal to shoot | R = reset | {mode}",
            True,
            WHITE
        )

        screen.blit(
            text,
            (20, 20)
        )

        # Result overlay
        if result is not None:
            big_font = pygame.font.Font(None, 96)
            small_font = pygame.font.Font(None, 28)
            color = WHITE
            if result == "GOAL":
                color = YELLOW
            elif result == "SAVED":
                color = BLUE
            elif result == "MISSED":
                color = RED
            # shadow
            shadow = big_font.render(result, True, BLACK)
            label = big_font.render(result, True, color)
            rx = WIDTH // 2 - label.get_width() // 2
            ry = HEIGHT // 2 - label.get_height() // 2
            screen.blit(shadow, (rx + 3, ry + 3))
            screen.blit(label, (rx, ry))
            hint = small_font.render("resetting...", True, WHITE)
            screen.blit(hint, (WIDTH // 2 - hint.get_width() // 2, ry + label.get_height() + 8))

        pygame.display.flip()

        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()