import pygame
import numpy as np
import math
from agent import Agent
from parked_car import ParkedCar
from car import Car
import time
import os
class Environment:
    def __init__(self):
        pygame.init()

        self.screen_width = 400
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.bg_color = (230, 230, 230)
        self.car = Car(self.screen, self.screen_width / 2 + 40 , self.screen_height - 250)

        self.parked_car1 = ParkedCar(self.screen, 340, 195, -90)
        self.parked_car2 = ParkedCar(self.screen, 340, 335, -90)
        self.parked_car3 = ParkedCar(self.screen, 340, 405, -90)
        self.parked_car4 = ParkedCar(self.screen, 60, 195, 90)
        self.parked_car5 = ParkedCar(self.screen, 60, 265, 90)
        self.parked_car6 = ParkedCar(self.screen, 60, 335, 90)
        self.parked_car7 = ParkedCar(self.screen, 60, 405, 90)

        self.P0 = (240, 350)
        self.P1 = (240, 290)
        self.P2 = (280, 270)
        self.P3 = (315, 260)
        self.P4 = (315, 240)

        pygame.font.init()

    def draw(self):
        self.screen.fill(self.bg_color)

        # Drawing lanes and parking spaces
        lane_width = 80
        lane_height = self.screen_height
        space_width = 120
        space_height = 70

        lane_color = (100, 100, 100)
        space_color = (92, 122, 171)
        line_color = (255, 255, 255)
        border_color = (255, 255, 0)
        target_color = (60, 207, 43)
        wrong_parking_space_color = (255, 165, 0)  # Orange color

        left_empty_space = pygame.Rect(
            0, 0, (self.screen_width / 2) - lane_width, self.screen_height
        )
        pygame.draw.rect(self.screen, (0, 78, 0), left_empty_space)
        right_empty_space = pygame.Rect(
            (self.screen_width / 2) + lane_width,
            0,
            (self.screen_width / 2) - lane_width,
            self.screen_height,
        )
        pygame.draw.rect(self.screen, (0, 78, 0), right_empty_space)

        pygame.draw.rect(
            self.screen,
            lane_color,
            ((self.screen_width / 2) - (lane_width), 0, lane_width, lane_height),
        )
        pygame.draw.rect(
            self.screen,
            lane_color,
            ((self.screen_width / 2), 0, lane_width, lane_height),
        )

        line_height = 20
        line_spacing = 10
        num_lines = int(lane_height / (line_height + line_spacing))
        line_y = (self.screen_height - num_lines * (line_height + line_spacing)) / 2
        for i in range(num_lines):
            line_rect = pygame.Rect(
                (self.screen_width / 2) - 1.5, line_y, 3, line_height
            )
            pygame.draw.rect(self.screen, line_color, line_rect)
            line_y += line_height + line_spacing

        num_spaces = 4
        space_x = (self.screen_width / 2) + lane_width
        space_y = (self.screen_height - num_spaces * (space_height)) / 2
        for i in range(num_spaces):
            parking_space_rect = pygame.Rect(
                space_x, space_y, space_width, space_height
            )
            pygame.draw.rect(self.screen, space_color, parking_space_rect)
            if i == 1:
                target_space_rect = parking_space_rect
            else:
                pygame.draw.rect(self.screen, border_color, parking_space_rect, 2)
            space_y += space_height - 2

        pygame.draw.rect(self.screen, target_color, target_space_rect, 4)

        space_x = (self.screen_width / 2) - lane_width - space_width
        space_y = (self.screen_height - num_spaces * (space_height)) / 2
        for i in range(num_spaces):
            parking_space_rect = pygame.Rect(space_x, space_y, space_width, space_height)
            pygame.draw.rect(self.screen, space_color, parking_space_rect)
            pygame.draw.rect(self.screen, border_color, parking_space_rect, 2)
            space_y += space_height - 2

        for i in range(1, 8):
            parked_car = getattr(self, "parked_car" + str(i))
            parked_car.draw()
            # pygame.draw.rect(self.screen, (255, 0, 0), parked_car.rect, 2)  # Draw red line around each parked car

        self.car.draw()

        # self.draw_line_to_target()
        self.draw_parking_box()
        '''adding some code'''
        # in_right_parking_space_rect = pygame.Rect(340, 255, 30, 20)
        # pygame.draw.rect(self.screen, (255, 0, 0), in_right_parking_space_rect, 2)
        
        
        pygame.display.flip()



    # def draw_line_to_target(self):
    #     car_midpoint_x, car_midpoint_y = self.car.x, self.car.y
    #     pygame.draw.line(
    #         self.screen, (0, 0, 255), (car_midpoint_x, car_midpoint_y), (340, 265), 5
    #     )

    def bezier_point(self, t, P0, P1, P2, P3, P4):
        if t < 0.5:
            t_scaled = t * 2
            x = (
                (1 - t_scaled) ** 2 * P0[0]
                + 2 * (1 - t_scaled) * t_scaled * P1[0]
                + t_scaled**2 * P2[0]
            )
            y = (
                (1 - t_scaled) ** 2 * P0[1]
                + 2 * (1 - t_scaled) * t_scaled * P1[1]
                + t_scaled**2 * P2[1]
            )
        else:
            t_scaled = (t - 0.5) * 2
            x = (
                (1 - t_scaled) ** 2 * P2[0]
                + 2 * (1 - t_scaled) * t_scaled * P3[0]
                + t_scaled**2 * P4[0]
            )
            y = (
                (1 - t_scaled) ** 2 * P2[1]
                + 2 * (1 - t_scaled) * t_scaled * P3[1]
                + t_scaled**2 * P4[1]
            )
        return (x, y)

    def draw_parking_box(self):
        center_x, center_y = 340, 265
        width, height = 40, 30

        # Calculate top-left corner of the box
        x = center_x - width // 2
        y = center_y - height // 2
        parking_box_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        parking_box_color = (255, 255, 255, 128)  # RGBA color with alpha for transparency
        border_thickness = 2
        # pygame.draw.rect(
        #     parking_box_surface,
        #     parking_box_color,
        #     (0, 0, width, height),
        #     border_thickness,
        # )


        self.screen.blit(parking_box_surface, (x, y))

    def distance_to_parking_spot(self, car_position, parking_spot_position):
        return np.linalg.norm(np.array(car_position) - np.array(parking_spot_position))

    def reset(self):
        self.car.x = self.screen_width / 2 + 40
        self.car.y = self.screen_height - 250
        self.car.angle = -10

        state = np.array(
            [self.car.x, self.car.y, self.car.acceleration, self.car.angle]
        )
        return state

    def distance_to_bezier(self, x, y):
        num_points = 100
        min_distance = float("inf")

        for i in range(num_points):
            t = i / (num_points - 1)
            point = self.bezier_point(t, self.P0, self.P1, self.P2, self.P3, self.P4)
            distance = math.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)
            if distance < min_distance:
                min_distance = distance

        return min_distance

    def step(self, action):
        if action == 0:
            acceleration = 1
            angle = self.car.angle
        elif action == 1:
            acceleration = -1
            angle = self.car.angle
        elif action == 2:
            acceleration = 0
            angle = self.car.angle + 10
        elif action == 3:
            acceleration = 0
            angle = self.car.angle - 10
        elif action == 4:
            acceleration = 0
            angle = self.car.angle

        self.car.acceleration = acceleration
        self.car.angle = angle
        self.car.update()
        
        
        boundary_hit = self.car.handle_boundary_perpendicular()
        
        state = np.array([self.car.x, self.car.y, self.car.acceleration, self.car.angle])
        target_x, target_y = 340, 265

        prev_distance = math.sqrt((self.car.prev_x - target_x) ** 2 + (self.car.prev_y - target_y) ** 2)
        distance = math.sqrt((self.car.x - target_x) ** 2 + (self.car.y - target_y) ** 2)

        bezier_distance = self.distance_to_bezier(self.car.x, self.car.y)
        
        # car_rect = self.car.get_bounding_box()
        car_collision = False
        # collision_rect = None
        for i in range(1, 8):
            parked_car = getattr(self, "parked_car" + str(i))
            if self.car.check_collision(parked_car.rect):
                car_collision = True
                #collision_rect = car_rect.clip(parked_car.rect)
                break

        in_lane = 215 <= self.car.x
        
        in_right_parking_space = (
            (self.car.x >= 340)
            and (self.car.x <= 350)
            and (self.car.y >= 255)
            and (self.car.y <= 275)
            # and (85 <= abs(self.car.angle % 360) <= 105)
        )
        
        target_dir = math.atan2(target_y - self.car.y, target_x - self.car.x)
        direction_diff = abs(target_dir - math.radians(self.car.angle))
        direction_diff = ((direction_diff + math.pi) % (2 * math.pi)) - math.pi

        p = 40000
        crash_penalty = -300
        time_penalty = -30
        movement_penalty = -20
        distance_reward_scale = 50
        orientation_reward_scale = 20
        efficiency_penalty = -50

        reward = 0
        done = False

        distance_reward = max(0, prev_distance - distance) * distance_reward_scale
        proximity_reward = (1 / (distance + 1)) * 100  # Higher reward as the distance decreases
        reward += distance_reward + proximity_reward

        reward += orientation_reward_scale * (1 - abs(direction_diff / math.pi))

        if distance < prev_distance:
            reward += movement_penalty
        else:
            reward -= efficiency_penalty

        if in_right_parking_space:
            reward = p
            print("parked")
            time.sleep(5)
            done = True

        elif boundary_hit or car_collision:
            reward = crash_penalty
            print("collided")
            done = True

        elif bezier_distance > 20:
            reward -= 0.5
        elif direction_diff > 0.5:
            reward -= 0.5
        else:
            reward += time_penalty

        # if collision_rect is not None:
        #     pygame.draw.rect(self.screen, (0, 0, 0), collision_rect,10)
        #     pygame.display.flip()  # Update the display to show the collision rectangle
        #     pygame.time.delay(500)  # Pause execution for 2000 milliseconds (2 seconds)
        
        return state, reward, done


    def render(self, mode="human"):
        self.draw()

    def run(self):
        clock = pygame.time.Clock()
        fps = 30
        running = True
        episode = 0
        agent = Agent(state_size=4, action_size=5, seed=42)  # Initialize the deep Q-learning agent\
            
            
        if os.path.exists('agent_perpend.pth'):
            agent.load_model("agent_perpend.pth")
            
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # agent.save_model("agent_perpend.pth")
                    running = False

            state = self.reset()
            total_reward = 0
            done = False

            while not done:
                self.render()
                action = agent.act(state)
                next_state, reward, done = self.step(action)
                total_reward += reward
                #agent.step(state, action, reward, next_state, done)  # Step through the agent's learning process
                state = next_state
                clock.tick(fps)

            print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
            print(self.car.angle)
            episode += 1

        pygame.quit()

if __name__ == "__main__":
    env = Environment()
    env.run()
