import pygame
import math
import numpy as np
from typing import Tuple, List, Optional

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (220, 50, 50)
GREEN = (50, 200, 80)
BLUE = (40, 120, 220)
YELLOW = (255, 210, 50)
GRAY = (60, 60, 70)
DARK_GRAY = (35, 35, 45)
LIGHT_GRAY = (180, 180, 190)
CYAN = (0, 220, 220)
ORANGE = (255, 140, 0)
TRACK_COLOR = (50, 50, 60)
GRASS_COLOR = (30, 85, 40)
BORDER_COLOR = (200, 200, 210)

WINDOW_WIDTH: int = 1400
WINDOW_HEIGHT: int = 900
FPS: int = 60

CAR_WIDTH: int = 20
CAR_HEIGHT: int = 40
MAX_SPEED: float = 8.0
ACCELERATION: float = 0.3
BRAKE_FORCE: float = 0.5
FRICTION: float = 0.05
TURN_RATE: float = 4.0
SENSOR_LENGTH: int = 200
NUM_SENSORS: int = 9
NUM_CARS: int = 200

CAR_COLORS = [
    (220, 50, 50), (50, 200, 80), (40, 120, 220), (255, 210, 50),
    (255, 140, 0), (180, 50, 220), (50, 220, 220), (220, 120, 180),
    (140, 200, 60), (60, 180, 200), (200, 80, 80), (80, 200, 140),
    (100, 100, 220), (220, 180, 60), (200, 100, 50), (120, 60, 200),
    (60, 200, 160), (200, 160, 120), (160, 200, 80), (80, 140, 200),
]


class RacingTrack:

    def __init__(self) -> None:
        cx, cy = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2

        self.outer_points: List[Tuple[int, int]] = [
            (cx - 650, cy + 50),
            (cx - 640, cy - 20),
            (cx - 580, cy - 80),
            (cx - 450, cy - 120),
            (cx - 200, cy - 130),
            (cx + 100, cy - 120),
            (cx + 350, cy - 150),
            (cx + 550, cy - 250),
            (cx + 640, cy - 150),
            (cx + 620, cy + 20),
            (cx + 550, cy + 120),
            (cx + 400, cy + 180),
            (cx + 200, cy + 150),
            (cx + 50, cy + 120),
            (cx - 50, cy + 180),
            (cx - 200, cy + 250),
            (cx - 400, cy + 280),
            (cx - 550, cy + 220),
            (cx - 620, cy + 150),
        ]

        self.inner_points: List[Tuple[int, int]] = [
            (cx - 540, cy + 50),
            (cx - 530, cy + 20),
            (cx - 470, cy - 20),
            (cx - 340, cy - 30),
            (cx - 110, cy - 40),
            (cx + 100, cy - 20),
            (cx + 280, cy - 50),
            (cx + 420, cy - 120),
            (cx + 510, cy - 50),
            (cx + 500, cy + 20),
            (cx + 440, cy + 40),
            (cx + 310, cy + 90),
            (cx + 130, cy + 60),
            (cx - 10, cy + 30),
            (cx - 50, cy + 80),
            (cx - 120, cy + 160),
            (cx - 310, cy + 190),
            (cx - 450, cy + 140),
            (cx - 520, cy + 90),
        ]

        self._build_walls()
        self._build_checkpoints()

    def _build_walls(self) -> None:
        self.walls: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        for i in range(len(self.outer_points)):
            p1 = self.outer_points[i]
            p2 = self.outer_points[(i + 1) % len(self.outer_points)]
            self.walls.append((p1, p2))
        for i in range(len(self.inner_points)):
            p1 = self.inner_points[i]
            p2 = self.inner_points[(i + 1) % len(self.inner_points)]
            self.walls.append((p1, p2))

    def _build_checkpoints(self) -> None:
        self.checkpoints: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        n_outer = len(self.outer_points)
        n_inner = len(self.inner_points)
        num_checkpoints = min(n_outer, n_inner)

        for i in range(num_checkpoints):
            oi = i % n_outer
            ii = i % n_inner
            self.checkpoints.append((self.outer_points[oi], self.inner_points[ii]))

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(GRASS_COLOR)

        pygame.draw.polygon(surface, TRACK_COLOR, self.outer_points)
        pygame.draw.polygon(surface, GRASS_COLOR, self.inner_points)

        pygame.draw.polygon(surface, BORDER_COLOR, self.outer_points, 3)
        pygame.draw.polygon(surface, BORDER_COLOR, self.inner_points, 3)

        if len(self.checkpoints) > 0:
            cp = self.checkpoints[0]
            pygame.draw.line(surface, YELLOW, cp[0], cp[1], 3)


class Car:

    def __init__(self, x: float, y: float, angle: float = 90.0, car_id: int = 0) -> None:
        self.start_x: float = x
        self.start_y: float = y
        self.start_angle: float = angle
        self.car_id: int = car_id

        self.x: float = x
        self.y: float = y
        self.angle: float = angle
        self.speed: float = 0.0
        self.alive: bool = True

        self.distance_traveled: float = 0.0
        self.time_alive: int = 0
        self.current_checkpoint: int = 0
        self.checkpoints_passed: int = 0

        self.sensor_angles: List[float] = [-90, -67.5, -45, -22.5, 0, 22.5, 45, 67.5, 90]
        self.sensor_readings: List[float] = [1.0] * NUM_SENSORS
        self.prev_x: float = x
        self.prev_y: float = y
        self.idle_steps: int = 0
        self.last_checkpoint_step: int = 0

    def reset(self) -> None:
        self.x = self.start_x
        self.y = self.start_y
        self.angle = self.start_angle
        self.speed = 0.0
        self.alive = True
        self.distance_traveled = 0.0
        self.time_alive = 0
        self.current_checkpoint = 0
        self.checkpoints_passed = 0
        self.sensor_readings = [1.0] * NUM_SENSORS
        self.prev_x = self.start_x
        self.prev_y = self.start_y
        self.idle_steps = 0
        self.last_checkpoint_step = 0

    def update(self, action: int) -> None:
        if not self.alive:
            return

        if action == 0:
            self.speed += ACCELERATION
        elif action == 1:
            self.speed -= BRAKE_FORCE
        elif action == 2:
            self.angle += TURN_RATE * max(0.5, self.speed / MAX_SPEED)
            self.speed += ACCELERATION * 0.7
        elif action == 3:
            self.angle -= TURN_RATE * max(0.5, self.speed / MAX_SPEED)
            self.speed += ACCELERATION * 0.7
        elif action == 4:
            pass

        if self.speed > 0:
            self.speed -= FRICTION
        elif self.speed < 0:
            self.speed += FRICTION

        self.speed = max(0.0, min(MAX_SPEED, self.speed))

        if abs(self.speed) < 0.01:
            self.speed = 0.0

        rad = math.radians(self.angle)
        dx = self.speed * math.cos(rad)
        dy = -self.speed * math.sin(rad)
        self.x += dx
        self.y += dy

        self.distance_traveled += abs(self.speed)
        self.time_alive += 1

        actual_move = math.hypot(self.x - self.prev_x, self.y - self.prev_y)
        if actual_move < 1.0:
            self.idle_steps += 1
        else:
            self.idle_steps = max(0, self.idle_steps - 2)
        self.prev_x = self.x
        self.prev_y = self.y

    def cast_sensors(
        self, walls: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    ) -> List[float]:
        self.sensor_readings = []
        self.sensor_endpoints: List[Tuple[float, float]] = []

        for s_angle in self.sensor_angles:
            total_angle = math.radians(self.angle + s_angle)
            end_x = self.x + SENSOR_LENGTH * math.cos(total_angle)
            end_y = self.y - SENSOR_LENGTH * math.sin(total_angle)

            closest_dist = SENSOR_LENGTH
            closest_point = (end_x, end_y)

            for wall in walls:
                intersection = self._line_intersection(
                    (self.x, self.y), (end_x, end_y), wall[0], wall[1]
                )
                if intersection is not None:
                    dist = math.hypot(
                        intersection[0] - self.x, intersection[1] - self.y
                    )
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_point = intersection

            self.sensor_readings.append(closest_dist / SENSOR_LENGTH)
            self.sensor_endpoints.append(closest_point)

        return self.sensor_readings

    def check_collision(
        self, walls: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    ) -> bool:
        corners = self._get_corners()
        for i in range(len(corners)):
            p1 = corners[i]
            p2 = corners[(i + 1) % len(corners)]
            for wall in walls:
                if self._line_intersection(p1, p2, wall[0], wall[1]) is not None:
                    self.alive = False
                    return True
        return False

    def check_checkpoint(
        self,
        checkpoints: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    ) -> bool:
        if not checkpoints:
            return False

        target = self.current_checkpoint % len(checkpoints)
        cp = checkpoints[target]

        corners = self._get_corners()
        for i in range(len(corners)):
            p1 = corners[i]
            p2 = corners[(i + 1) % len(corners)]
            if self._line_intersection(p1, p2, cp[0], cp[1]) is not None:
                self.current_checkpoint += 1
                self.checkpoints_passed += 1
                return True
        return False

    def _get_corners(self) -> List[Tuple[float, float]]:
        rad = math.radians(self.angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        hw, hh = CAR_WIDTH / 2, CAR_HEIGHT / 2

        return [
            (self.x + cos_a * hh - sin_a * hw, self.y - sin_a * hh - cos_a * hw),
            (self.x + cos_a * hh + sin_a * hw, self.y - sin_a * hh + cos_a * hw),
            (self.x - cos_a * hh + sin_a * hw, self.y + sin_a * hh + cos_a * hw),
            (self.x - cos_a * hh - sin_a * hw, self.y + sin_a * hh - cos_a * hw),
        ]

    @staticmethod
    def _line_intersection(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        p4: Tuple[float, float],
    ) -> Optional[Tuple[float, float]]:
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        if 0 <= t <= 1 and 0 <= u <= 1:
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            return (ix, iy)
        return None

    def get_state(self) -> List[float]:
        return self.sensor_readings + [
            self.speed / MAX_SPEED,
            math.sin(math.radians(self.angle)),
            math.cos(math.radians(self.angle)),
        ]

    def draw(self, surface: pygame.Surface, color: Tuple[int, int, int] = RED,
             draw_sensors: bool = False) -> None:
        if not self.alive:
            return

        if draw_sensors:
            for i, endpoint in enumerate(getattr(self, "sensor_endpoints", [])):
                reading = self.sensor_readings[i]
                r = int(255 * (1 - reading))
                g = int(255 * reading)
                sensor_color = (r, g, 50)
                pygame.draw.line(surface, sensor_color, (int(self.x), int(self.y)),
                                 (int(endpoint[0]), int(endpoint[1])), 1)
                pygame.draw.circle(surface, WHITE, (int(endpoint[0]), int(endpoint[1])), 3)

        corners = self._get_corners()
        int_corners = [(int(c[0]), int(c[1])) for c in corners]
        pygame.draw.polygon(surface, color, int_corners)
        pygame.draw.polygon(surface, WHITE, int_corners, 1)

        if draw_sensors:
            rad = math.radians(self.angle)
            front_x = self.x + (CAR_HEIGHT / 2 + 5) * math.cos(rad)
            front_y = self.y - (CAR_HEIGHT / 2 + 5) * math.sin(rad)
            pygame.draw.circle(surface, YELLOW, (int(front_x), int(front_y)), 4)


class RacingEnvironment:

    STATE_SIZE: int = 12
    ACTION_SIZE: int = 5

    def __init__(self, render: bool = True, num_cars: int = NUM_CARS) -> None:
        self.render_enabled: bool = render
        self.num_cars: int = num_cars
        self.track: RacingTrack = RacingTrack()

        cp0 = self.track.checkpoints[0]
        cp1 = self.track.checkpoints[1]
        self.start_x = (cp0[0][0] + cp0[1][0]) / 2
        self.start_y = (cp0[0][1] + cp0[1][1]) / 2

        next_x = (cp1[0][0] + cp1[1][0]) / 2
        next_y = (cp1[0][1] + cp1[1][1]) / 2
        start_angle = math.degrees(math.atan2(-(next_y - self.start_y), next_x - self.start_x))

        self.cars: List[Car] = [
            Car(self.start_x, self.start_y, angle=start_angle, car_id=i)
            for i in range(self.num_cars)
        ]

        self.car_steps: List[int] = [0] * self.num_cars
        self.car_rewards: List[float] = [0.0] * self.num_cars
        self.car_done: List[bool] = [False] * self.num_cars
        self.max_steps: int = 2000
        self.global_step: int = 0

        if self.render_enabled:
            pygame.init()
            self.screen: pygame.Surface = pygame.display.set_mode(
                (WINDOW_WIDTH, WINDOW_HEIGHT)
            )
            pygame.display.set_caption(f"Autonomous Racing Car - DQN Training ({self.num_cars} Cars)")
            self.clock: pygame.Clock = pygame.time.Clock()
            self.font: pygame.font.Font = pygame.font.SysFont("consolas", 18)
            self.font_large: pygame.font.Font = pygame.font.SysFont("consolas", 24, bold=True)

    def reset(self) -> List[List[float]]:
        states = []
        for i in range(self.num_cars):
            self.cars[i].reset()
            self.car_steps[i] = 0
            self.car_rewards[i] = 0.0
            self.car_done[i] = False
            self.cars[i].cast_sensors(self.track.walls)
            states.append(self.cars[i].get_state())
        self.global_step = 0
        return states

    def step(self, actions: List[int]) -> Tuple[List[List[float]], List[float], List[bool]]:
        self.global_step += 1
        states: List[List[float]] = []
        rewards: List[float] = []
        dones: List[bool] = []

        for i in range(self.num_cars):
            if self.car_done[i]:
                self.cars[i].reset()
                self.car_steps[i] = 0
                self.car_rewards[i] = 0.0
                self.car_done[i] = False
                self.cars[i].cast_sensors(self.track.walls)

            car = self.cars[i]
            self.car_steps[i] += 1
            prev_cp_dist = self._get_checkpoint_distance(car)

            car.update(actions[i])
            car.cast_sensors(self.track.walls)

            collision = car.check_collision(self.track.walls)
            checkpoint_crossed = car.check_checkpoint(self.track.checkpoints)

            reward = self._calculate_reward(car, collision, checkpoint_crossed, prev_cp_dist)

            done = collision or self.car_steps[i] >= self.max_steps
            self.car_done[i] = done
            self.car_rewards[i] += reward

            states.append(car.get_state())
            rewards.append(reward)
            dones.append(done)

        return states, rewards, dones

    def _get_checkpoint_distance(self, car: Car) -> float:
        target = car.current_checkpoint % len(self.track.checkpoints)
        cp = self.track.checkpoints[target]
        cp_x = (cp[0][0] + cp[1][0]) / 2
        cp_y = (cp[0][1] + cp[1][1]) / 2
        return math.hypot(car.x - cp_x, car.y - cp_y)

    def _calculate_reward(
        self,
        car: Car,
        collision: bool,
        checkpoint_crossed: bool,
        prev_cp_dist: float,
    ) -> float:
        if collision:
            return -10.0

        reward: float = 0.0

        if checkpoint_crossed:
            reward += 25.0
            car.last_checkpoint_step = car.time_alive

        curr_cp_dist = self._get_checkpoint_distance(car)
        dist_delta = prev_cp_dist - curr_cp_dist
        reward += dist_delta * 0.05

        if car.speed > 0.5:
            reward += (car.speed / MAX_SPEED) * 1.0
        else:
            reward -= 0.5

        if car.idle_steps > 30:
            reward -= 2.0
        elif car.idle_steps > 15:
            reward -= 1.0

        steps_since_cp = car.time_alive - car.last_checkpoint_step
        if steps_since_cp > 300:
            reward -= 1.5
        elif steps_since_cp > 150:
            reward -= 0.5

        min_sensor = min(car.sensor_readings)
        if min_sensor < 0.1:
            reward -= (0.1 - min_sensor) * 5.0

        return reward

    def _get_best_car_index(self) -> int:
        best_idx = 0
        best_score = -float("inf")
        for i in range(self.num_cars):
            score = self.cars[i].checkpoints_passed * 1000 + self.cars[i].distance_traveled
            if self.cars[i].alive:
                score += 10000
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def draw(
        self,
        episode: int = 0,
        high_score: float = 0.0,
        current_reward: float = 0.0,
        epsilon: float = 1.0,
        total_steps: int = 0,
    ) -> None:
        if not self.render_enabled:
            return

        self.track.draw(self.screen)

        best_idx = self._get_best_car_index()
        alive_count = sum(1 for c in self.cars if c.alive)

        for i in range(self.num_cars):
            if i == best_idx:
                continue
            color = CAR_COLORS[i % len(CAR_COLORS)]
            faded = (color[0] // 3, color[1] // 3, color[2] // 3)
            self.cars[i].draw(self.screen, color=faded, draw_sensors=False)

        self.cars[best_idx].draw(self.screen, color=CYAN, draw_sensors=True)

        self._draw_ui(episode, high_score, current_reward, epsilon, total_steps,
                       alive_count, best_idx)
        pygame.display.flip()
        self.clock.tick(FPS)

    def _draw_ui(
        self,
        episode: int,
        high_score: float,
        current_reward: float,
        epsilon: float,
        total_steps: int,
        alive_count: int,
        best_idx: int,
    ) -> None:
        panel = pygame.Surface((320, 310), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 180))
        self.screen.blit(panel, (10, 10))

        title = self.font_large.render("DQN Racing Agent", True, CYAN)
        self.screen.blit(title, (20, 18))

        pygame.draw.line(self.screen, CYAN, (20, 48), (310, 48), 1)

        best_car = self.cars[best_idx]
        best_reward = self.car_rewards[best_idx]

        stats = [
            (f"Episode:        {episode}", WHITE),
            (f"High Score:     {high_score:.1f}", YELLOW),
            (f"Best Reward:    {best_reward:.1f}", GREEN if best_reward > 0 else RED),
            (f"Avg Reward:     {current_reward:.1f}", GREEN if current_reward > 0 else RED),
            (f"Epsilon:        {epsilon:.4f}", ORANGE),
            (f"Cars Alive:     {alive_count}/{self.num_cars}", GREEN if alive_count > 25 else RED),
            (f"Best CP:        {best_car.checkpoints_passed}", WHITE),
            (f"Best Speed:     {best_car.speed:.1f}", WHITE),
            (f"Step:           {self.global_step}/{self.max_steps}", LIGHT_GRAY),
            (f"Total Steps:    {total_steps}", LIGHT_GRAY),
        ]

        y = 58
        for text, color in stats:
            surf = self.font.render(text, True, color)
            self.screen.blit(surf, (20, y))
            y += 24

        bar_panel = pygame.Surface((340, 80), pygame.SRCALPHA)
        bar_panel.fill((0, 0, 0, 180))
        self.screen.blit(bar_panel, (10, 330))

        label = self.font.render("Best Car Sensors", True, CYAN)
        self.screen.blit(label, (20, 338))

        labels = ["L90", "L67", "L45", "L22", "FWD", "R22", "R45", "R67", "R90"]
        for i, reading in enumerate(best_car.sensor_readings):
            bx = 18 + i * 35
            by = 390
            pygame.draw.rect(self.screen, DARK_GRAY, (bx, by - 30, 30, 30))
            fill_h = int(reading * 30)
            r = int(255 * (1 - reading))
            g = int(255 * reading)
            pygame.draw.rect(self.screen, (r, g, 50), (bx, by - fill_h, 30, fill_h))
            pygame.draw.rect(self.screen, WHITE, (bx, by - 30, 30, 30), 1)
            lbl = self.font.render(labels[i], True, WHITE)
            self.screen.blit(lbl, (bx, by + 2))

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True
