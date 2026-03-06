import os
import sys
import math
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F

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

WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
FPS = 60

CAR_WIDTH = 20
CAR_HEIGHT = 40
MAX_SPEED = 8.0
ACCELERATION = 0.3
BRAKE_FORCE = 0.5
FRICTION = 0.05
TURN_RATE = 4.0
SENSOR_LENGTH = 200
NUM_SENSORS = 9
NUM_CARS = 50

STATE_SIZE = NUM_SENSORS + 3
ACTION_SIZE = 5
MODEL_PATH = "model.pth"

CAR_COLORS = [
    (220, 50, 50), (50, 200, 80), (40, 120, 220), (255, 210, 50),
    (255, 140, 0), (180, 50, 220), (50, 220, 220), (220, 120, 180),
    (140, 200, 60), (60, 180, 200), (200, 80, 80), (80, 200, 140),
    (100, 100, 220), (220, 180, 60), (200, 100, 50), (120, 60, 200),
    (60, 200, 160), (200, 160, 120), (160, 200, 80), (80, 140, 200),
]


def line_intersect(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-10:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / d
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None


class NoisyLinear(nn.Module):

    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.sigma_init = sigma_init
        self._reset_parameters()
        self.reset_noise()

    def _reset_parameters(self):
        r = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-r, r)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-r, r)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        ei = self._scale_noise(self.in_features)
        eo = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eo.ger(ei))
        self.bias_epsilon.copy_(eo)

    def forward(self, x):
        if self.training:
            return F.linear(
                x,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon,
            )
        return F.linear(x, self.weight_mu, self.bias_mu)


class DQN(nn.Module):

    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.bn1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.LayerNorm(128)
        self.value_stream = NoisyLinear(128, 1)
        self.advantage_stream = NoisyLinear(128, action_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        v = self.value_stream(x)
        a = self.advantage_stream(x)
        return v + (a - a.mean(dim=1, keepdim=True))

    def reset_noise(self):
        self.value_stream.reset_noise()
        self.advantage_stream.reset_noise()


def _remap_notebook_keys(state_dict):
    mapping = {
        "val.": "value_stream.",
        "adv.": "advantage_stream.",
        "w_mu": "weight_mu",
        "w_sigma": "weight_sigma",
        "w_eps": "weight_epsilon",
        "b_mu": "bias_mu",
        "b_sigma": "bias_sigma",
        "b_eps": "bias_epsilon",
    }
    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        for old, new in mapping.items():
            nk = nk.replace(old, new)
        new_sd[nk] = v
    return new_sd


class RacingTrack:

    def __init__(self):
        cx, cy = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
        self.outer_points = [
            (cx - 650, cy + 50), (cx - 640, cy - 20), (cx - 580, cy - 80),
            (cx - 450, cy - 120), (cx - 200, cy - 130), (cx + 100, cy - 120),
            (cx + 350, cy - 150), (cx + 550, cy - 250), (cx + 640, cy - 150),
            (cx + 620, cy + 20), (cx + 550, cy + 120), (cx + 400, cy + 180),
            (cx + 200, cy + 150), (cx + 50, cy + 120), (cx - 50, cy + 180),
            (cx - 200, cy + 250), (cx - 400, cy + 280), (cx - 550, cy + 220),
            (cx - 620, cy + 150),
        ]
        self.inner_points = [
            (cx - 540, cy + 50), (cx - 530, cy + 20), (cx - 470, cy - 20),
            (cx - 340, cy - 30), (cx - 110, cy - 40), (cx + 100, cy - 20),
            (cx + 280, cy - 50), (cx + 420, cy - 120), (cx + 510, cy - 50),
            (cx + 500, cy + 20), (cx + 440, cy + 40), (cx + 310, cy + 90),
            (cx + 130, cy + 60), (cx - 10, cy + 30), (cx - 50, cy + 80),
            (cx - 120, cy + 160), (cx - 310, cy + 190), (cx - 450, cy + 140),
            (cx - 520, cy + 90),
        ]
        self.walls = []
        for pts in [self.outer_points, self.inner_points]:
            for i in range(len(pts)):
                self.walls.append((pts[i], pts[(i + 1) % len(pts)]))
        self.checkpoints = [
            (self.outer_points[i], self.inner_points[i])
            for i in range(len(self.outer_points))
        ]

    def draw(self, surface):
        surface.fill(GRASS_COLOR)
        pygame.draw.polygon(surface, TRACK_COLOR, self.outer_points)
        pygame.draw.polygon(surface, GRASS_COLOR, self.inner_points)
        pygame.draw.polygon(surface, BORDER_COLOR, self.outer_points, 3)
        pygame.draw.polygon(surface, BORDER_COLOR, self.inner_points, 3)
        if self.checkpoints:
            cp = self.checkpoints[0]
            pygame.draw.line(surface, YELLOW, cp[0], cp[1], 3)


class Car:

    def __init__(self, x, y, angle=90.0, car_id=0, start_cp=0):
        self.start_x = x
        self.start_y = y
        self.start_angle = angle
        self.car_id = car_id
        self.start_cp = start_cp
        self.sensor_angles = [-90, -67.5, -45, -22.5, 0, 22.5, 45, 67.5, 90]
        self.reset()

    def reset(self):
        self.x = self.start_x
        self.y = self.start_y
        self.angle = self.start_angle
        self.speed = 0.0
        self.alive = True
        self.distance_traveled = 0.0
        self.time_alive = 0
        self.current_checkpoint = self.start_cp
        self.checkpoints_passed = 0
        self.sensor_readings = [1.0] * NUM_SENSORS
        self.sensor_endpoints = []
        self.prev_x = self.start_x
        self.prev_y = self.start_y
        self.idle_steps = 0
        self.last_checkpoint_step = 0

    def update(self, action):
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
        if self.speed > 0:
            self.speed -= FRICTION
        elif self.speed < 0:
            self.speed += FRICTION
        self.speed = max(0.0, min(MAX_SPEED, self.speed))
        if abs(self.speed) < 0.01:
            self.speed = 0.0
        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y -= self.speed * math.sin(rad)
        self.distance_traveled += abs(self.speed)
        self.time_alive += 1
        actual_move = math.hypot(self.x - self.prev_x, self.y - self.prev_y)
        if actual_move < 1.0:
            self.idle_steps += 1
        else:
            self.idle_steps = max(0, self.idle_steps - 2)
        self.prev_x = self.x
        self.prev_y = self.y

    def cast_sensors(self, walls):
        self.sensor_readings = []
        self.sensor_endpoints = []
        for s_angle in self.sensor_angles:
            total_angle = math.radians(self.angle + s_angle)
            end_x = self.x + SENSOR_LENGTH * math.cos(total_angle)
            end_y = self.y - SENSOR_LENGTH * math.sin(total_angle)
            closest_dist = SENSOR_LENGTH
            closest_point = (end_x, end_y)
            for wall in walls:
                intersection = line_intersect(
                    (self.x, self.y), (end_x, end_y), wall[0], wall[1]
                )
                if intersection is not None:
                    dist = math.hypot(intersection[0] - self.x, intersection[1] - self.y)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_point = intersection
            self.sensor_readings.append(closest_dist / SENSOR_LENGTH)
            self.sensor_endpoints.append(closest_point)
        return self.sensor_readings

    def check_collision(self, walls):
        corners = self._get_corners()
        for i in range(4):
            for wall in walls:
                if line_intersect(corners[i], corners[(i + 1) % 4], wall[0], wall[1]):
                    self.alive = False
                    return True
        return False

    def check_checkpoint(self, checkpoints):
        if not checkpoints:
            return False
        target = self.current_checkpoint % len(checkpoints)
        cp = checkpoints[target]
        corners = self._get_corners()
        for i in range(4):
            if line_intersect(corners[i], corners[(i + 1) % 4], cp[0], cp[1]):
                self.current_checkpoint += 1
                self.checkpoints_passed += 1
                return True
        return False

    def _get_corners(self):
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

    def get_state(self):
        return self.sensor_readings + [
            self.speed / MAX_SPEED,
            math.sin(math.radians(self.angle)),
            math.cos(math.radians(self.angle)),
        ]

    def draw(self, surface, color=RED, draw_sensors=False):
        if not self.alive:
            return
        if draw_sensors:
            for i, endpoint in enumerate(self.sensor_endpoints):
                reading = self.sensor_readings[i]
                r = int(255 * (1 - reading))
                g = int(255 * reading)
                sensor_color = (r, g, 50)
                pygame.draw.line(
                    surface, sensor_color,
                    (int(self.x), int(self.y)),
                    (int(endpoint[0]), int(endpoint[1])), 1,
                )
                pygame.draw.circle(
                    surface, WHITE,
                    (int(endpoint[0]), int(endpoint[1])), 3,
                )
        corners = self._get_corners()
        int_corners = [(int(c[0]), int(c[1])) for c in corners]
        pygame.draw.polygon(surface, color, int_corners)
        pygame.draw.polygon(surface, WHITE, int_corners, 1)
        if draw_sensors:
            rad = math.radians(self.angle)
            front_x = self.x + (CAR_HEIGHT / 2 + 5) * math.cos(rad)
            front_y = self.y - (CAR_HEIGHT / 2 + 5) * math.sin(rad)
            pygame.draw.circle(surface, YELLOW, (int(front_x), int(front_y)), 4)


def load_model(path, device):
    net = DQN(STATE_SIZE, ACTION_SIZE).to(device)
    if not os.path.exists(path):
        print(f"No model found at {path}")
        return None, {}

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if "policy_net_state_dict" in checkpoint:
        net.load_state_dict(checkpoint["policy_net_state_dict"])
        info = {
            "episode": checkpoint.get("episode", 0),
            "high_score": checkpoint.get("high_score", 0),
            "training_step": checkpoint.get("training_step", 0),
        }
    elif "pnet" in checkpoint:
        remapped = _remap_notebook_keys(checkpoint["pnet"])
        net.load_state_dict(remapped)
        info = {
            "episode": checkpoint.get("ep", 0),
            "high_score": checkpoint.get("hi", 0),
            "training_step": checkpoint.get("tstep", 0),
        }
    else:
        try:
            net.load_state_dict(checkpoint)
            info = {}
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None, {}

    net.eval()
    print(f"Model loaded from {path}")
    if info:
        print(f"  Episode: {info.get('episode', '?')}, "
              f"High Score: {info.get('high_score', '?')}, "
              f"Steps: {info.get('training_step', '?')}")
    return net, info


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    model_path = MODEL_PATH
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    net, info = load_model(model_path, device)
    if net is None:
        print("Cannot run without a trained model.")
        return

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Autonomous Racing Car - Visualization")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    font_large = pygame.font.SysFont("consolas", 24, bold=True)

    track = RacingTrack()
    ncp = len(track.checkpoints)

    cars = []
    cp0 = track.checkpoints[0]
    ox, oy = cp0[0]
    ix, iy = cp0[1]
    cp1 = track.checkpoints[1]
    tx = (cp1[0][0] + cp1[1][0]) / 2
    ty = (cp1[0][1] + cp1[1][1]) / 2
    for i in range(NUM_CARS):
        t = (i + 1) / (NUM_CARS + 1)
        x = ox + t * (ix - ox)
        y = oy + t * (iy - oy)
        ang = math.degrees(math.atan2(-(ty - y), tx - x))
        cars.append(Car(x, y, ang, i, 1))

    car_rewards = [0.0] * NUM_CARS
    car_done = [False] * NUM_CARS
    car_steps = [0] * NUM_CARS
    max_steps = 2000
    global_step = 0
    episode = 1
    best_ever = info.get("high_score", 0)
    speed_multiplier = 1

    def reset_all():
        nonlocal global_step
        for i in range(NUM_CARS):
            cars[i].reset()
            car_rewards[i] = 0.0
            car_done[i] = False
            car_steps[i] = 0
            cars[i].cast_sensors(track.walls)
        global_step = 0

    reset_all()

    running = True
    paused = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    episode += 1
                    reset_all()
                elif event.key == pygame.K_UP:
                    speed_multiplier = min(speed_multiplier + 1, 10)
                elif event.key == pygame.K_DOWN:
                    speed_multiplier = max(speed_multiplier - 1, 1)

        if not paused:
            for _ in range(speed_multiplier):
                global_step += 1

                states = []
                alive_indices = []
                for i in range(NUM_CARS):
                    if car_done[i]:
                        cars[i].reset()
                        car_steps[i] = 0
                        car_rewards[i] = 0.0
                        car_done[i] = False
                        cars[i].cast_sensors(track.walls)
                    states.append(cars[i].get_state())
                    alive_indices.append(i)

                state_tensor = torch.tensor(states, dtype=torch.float32).to(device)
                with torch.no_grad():
                    q_values = net(state_tensor)
                actions = q_values.argmax(dim=1).tolist()

                for idx, i in enumerate(alive_indices):
                    car = cars[i]
                    car_steps[i] += 1
                    car.update(actions[idx])
                    car.cast_sensors(track.walls)
                    collision = car.check_collision(track.walls)
                    checkpoint_crossed = car.check_checkpoint(track.checkpoints)

                    reward = 0.0
                    if collision:
                        reward = -10.0
                    else:
                        if checkpoint_crossed:
                            reward += 25.0
                            car.last_checkpoint_step = car.time_alive
                        if car.speed > 0.5:
                            reward += (car.speed / MAX_SPEED) * 1.0

                    steps_since_cp = car.time_alive - car.last_checkpoint_step
                    if steps_since_cp > 500:
                        car.alive = False

                    done = collision or car_steps[i] >= max_steps or steps_since_cp > 500
                    car_done[i] = done
                    car_rewards[i] += reward

                if global_step >= max_steps:
                    best_ep = max(car_rewards)
                    best_ever = max(best_ever, best_ep)
                    episode += 1
                    reset_all()

        track.draw(screen)

        best_idx = 0
        best_score = -float("inf")
        alive_count = 0
        for i in range(NUM_CARS):
            if cars[i].alive:
                alive_count += 1
            score = cars[i].checkpoints_passed * 1000 + cars[i].distance_traveled
            if cars[i].alive:
                score += 10000
            if score > best_score:
                best_score = score
                best_idx = i

        for i in range(NUM_CARS):
            if i == best_idx:
                continue
            color = CAR_COLORS[i % len(CAR_COLORS)]
            faded = (color[0] // 3, color[1] // 3, color[2] // 3)
            cars[i].draw(screen, color=faded, draw_sensors=False)

        cars[best_idx].draw(screen, color=CYAN, draw_sensors=True)

        panel = pygame.Surface((320, 310), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 180))
        screen.blit(panel, (10, 10))

        title = font_large.render("DQN Racing - Eval", True, CYAN)
        screen.blit(title, (20, 18))
        pygame.draw.line(screen, CYAN, (20, 48), (310, 48), 1)

        best_car = cars[best_idx]
        best_reward = car_rewards[best_idx]
        avg_reward = sum(car_rewards) / NUM_CARS

        stats = [
            (f"Episode:        {episode}", WHITE),
            (f"High Score:     {best_ever:.1f}", YELLOW),
            (f"Best Reward:    {best_reward:.1f}", GREEN if best_reward > 0 else RED),
            (f"Avg Reward:     {avg_reward:.1f}", GREEN if avg_reward > 0 else RED),
            (f"Cars Alive:     {alive_count}/{NUM_CARS}", GREEN if alive_count > 10 else RED),
            (f"Best CP:        {best_car.checkpoints_passed}", WHITE),
            (f"Best Speed:     {best_car.speed:.1f}", WHITE),
            (f"Step:           {global_step}/{max_steps}", LIGHT_GRAY),
            (f"Speed:          {speed_multiplier}x", ORANGE),
            (f"Trained Steps:  {info.get('training_step', '?')}", LIGHT_GRAY),
        ]

        y = 58
        for text, color in stats:
            surf = font.render(text, True, color)
            screen.blit(surf, (20, y))
            y += 24

        bar_panel = pygame.Surface((340, 80), pygame.SRCALPHA)
        bar_panel.fill((0, 0, 0, 180))
        screen.blit(bar_panel, (10, 330))

        label = font.render("Best Car Sensors", True, CYAN)
        screen.blit(label, (20, 338))

        labels = ["L90", "L67", "L45", "L22", "FWD", "R22", "R45", "R67", "R90"]
        for i, reading in enumerate(best_car.sensor_readings):
            bx = 18 + i * 35
            by = 390
            pygame.draw.rect(screen, DARK_GRAY, (bx, by - 30, 30, 30))
            fill_h = int(reading * 30)
            r = int(255 * (1 - reading))
            g = int(255 * reading)
            pygame.draw.rect(screen, (r, g, 50), (bx, by - fill_h, 30, fill_h))
            pygame.draw.rect(screen, WHITE, (bx, by - 30, 30, 30), 1)
            lbl = font.render(labels[i], True, WHITE)
            screen.blit(lbl, (bx, by + 2))

        help_panel = pygame.Surface((260, 110), pygame.SRCALPHA)
        help_panel.fill((0, 0, 0, 180))
        screen.blit(help_panel, (WINDOW_WIDTH - 270, 10))

        controls = [
            ("Controls", CYAN),
            ("SPACE  - Pause/Resume", LIGHT_GRAY),
            ("R      - Reset episode", LIGHT_GRAY),
            ("UP/DN  - Speed +/-", LIGHT_GRAY),
            ("ESC    - Quit", LIGHT_GRAY),
        ]
        cy_text = 18
        for text, color in controls:
            surf = font.render(text, True, color)
            screen.blit(surf, (WINDOW_WIDTH - 260, cy_text))
            cy_text += 20

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
