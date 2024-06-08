import pygame
import math

WIDTH_BIAS = 25
HEIGHT_BIAS = 45

class Car:
    def __init__(self, screen, x, y):
        self.screen = screen
        self.width = 50
        self.height = 90
        self.car_image = pygame.image.load("assets/car.png")
        self.car_image = pygame.transform.scale(self.car_image, (self.width, self.height))
        self.x = x + self.width / 2
        self.y = y + self.height / 2
        self.speed = 1
        self.acceleration = 1
        self.max_speed = 5
        self.min_speed = -2
        self.angle = 0 % 360
        self.friction = 0.01

    def update(self):
        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_angle = self.angle

        if self.speed > 0:
            self.speed -= self.friction * self.speed
            if self.speed < 0:
                self.speed = 0
        elif self.speed < 0:
            self.speed += self.friction * abs(self.speed)
            if self.speed > 0:
                self.speed = 0

        self.speed += self.acceleration
        if self.speed > self.max_speed:
            self.speed = self.max_speed
        elif self.speed < self.min_speed:
            self.speed = self.min_speed

        angle_radians = math.radians(self.angle)
        new_x = self.x + self.speed * math.sin(-angle_radians)
        new_y = self.y - self.speed * math.cos(angle_radians)

        self.x = new_x
        self.y = new_y

    def draw(self):
        rotated_image = pygame.transform.rotate(self.car_image, self.angle)
        rect = rotated_image.get_rect(center=(self.x, self.y))
        self.screen.blit(rotated_image, rect)
        # self.get_bounding_box()

    # def get_bounding_box(self):
        # corners = self.get_rotated_corners()
        # for i in range(len(corners)):
            # pygame.draw.line(self.screen, (0, 0, 0),  corners[(i+1) % len(corners)], corners[i], 1)

    def get_rotated_corners(self):
        angle_radians = math.radians(self.angle)
        cos_theta = math.cos(angle_radians)
        sin_theta = -math.sin(angle_radians)

        half_width = self.width / 2
        half_height = self.height / 2

        corners = [
            (self.x + half_width * cos_theta - half_height * sin_theta, self.y + half_width * sin_theta + half_height * cos_theta),
            (self.x - half_width * cos_theta - half_height * sin_theta, self.y - half_width * sin_theta + half_height * cos_theta),
            (self.x - half_width * cos_theta + half_height * sin_theta, self.y - half_width * sin_theta - half_height * cos_theta),
            (self.x + half_width * cos_theta + half_height * sin_theta, self.y + half_width * sin_theta - half_height * cos_theta),
        ]
        return corners

    def check_collision(self, other_rect):
        def project_polygon(corners, axis):
            min_proj = float('inf')
            max_proj = float('-inf')
            for corner in corners:
                projection = corner[0] * axis[0] + corner[1] * axis[1]
                if projection < min_proj:
                    min_proj = projection
                if projection > max_proj:
                    max_proj = projection
            return min_proj, max_proj

        def overlap(min1, max1, min2, max2):
            return max1 >= min2 and max2 >= min1

        def get_axes(corners):
            axes = []
            for i in range(len(corners)):
                p1 = corners[i]
                p2 = corners[(i + 1) % len(corners)]
                edge = (p2[0] - p1[0], p2[1] - p1[1])
                normal = (-edge[1], edge[0])
                length = math.sqrt(normal[0]**2 + normal[1]**2)
                axes.append((normal[0] / length, normal[1] / length))
            return axes

        car_corners = self.get_rotated_corners()
        rect_corners = [
            (other_rect.left, other_rect.top),
            (other_rect.left, other_rect.bottom),
            (other_rect.right, other_rect.top),
            (other_rect.right, other_rect.bottom)
        ]

        car_axes = get_axes(car_corners)
        rect_axes = get_axes(rect_corners)

        for axis in car_axes + rect_axes:
            car_proj = project_polygon(car_corners, axis)
            rect_proj = project_polygon(rect_corners, axis)
            if not overlap(car_proj[0], car_proj[1], rect_proj[0], rect_proj[1]):
                return False
        return True
    
    def handle_boundary(self):
        new_x = self.x
        new_y = self.y
        collided = False
        if new_y < 60 or new_y > 540:  # (y-45) Since y is y-height/2
            if new_x < 120 + WIDTH_BIAS:
                new_x = 120 + WIDTH_BIAS
                collided = True
            if new_x > 280 - WIDTH_BIAS:
                new_x = 280 - WIDTH_BIAS
                collided = True
        else:
            if (60 <= new_x <= 125 or 280 <= new_x <= 315) and new_y < 60 + HEIGHT_BIAS:
                new_y = 60 + HEIGHT_BIAS
                collided = True
            if (
                60 <= new_x <= 125 or 280 <= new_x <= 315
            ) and new_y > 540 - HEIGHT_BIAS:
                new_y = 540 - HEIGHT_BIAS
                collided = True
            if new_x < 85:
                new_x = 85
                collided = True
            if new_x > 340 - WIDTH_BIAS:
                new_x = 340 - WIDTH_BIAS
                collided = True
        if new_y - HEIGHT_BIAS < 0:
            new_y = HEIGHT_BIAS
            collided = True
        if new_y + HEIGHT_BIAS > self.screen.get_height():
            new_y = self.screen.get_height() - HEIGHT_BIAS
            collided = True
        self.x = new_x
        self.y = new_y
        return collided

    def handle_boundary_perpendicular(self):
        new_x = self.x
        new_y = self.y

        collided = False

        if new_y < 180 or new_y > 420: # (y-45) Since y is y-height/2
            if new_x < 120 + WIDTH_BIAS:
                new_x = 120 + WIDTH_BIAS
                collided = True
            if new_x > 280 - WIDTH_BIAS:
                new_x = 280 - WIDTH_BIAS
                collided = True
        else:
            if (0 <= new_x <= 140 or 290 <= new_x <= 400) and new_y < 170 + HEIGHT_BIAS:
                new_y = 170 + HEIGHT_BIAS
                collided = True
            if (0 <= new_x <= 140 or 290 <= new_x <= 400) and new_y > 420 - HEIGHT_BIAS:
                new_y = 420 - HEIGHT_BIAS
                collided = True   

            if new_x < 45:
                new_x = 45
                collided = True
            if new_x > 380 - WIDTH_BIAS:
                new_x = 380 - WIDTH_BIAS
                collided = True

        if new_y - HEIGHT_BIAS < 0:
            new_y = HEIGHT_BIAS
            collided = True
        if new_y + HEIGHT_BIAS > self.screen.get_height():
            new_y = self.screen.get_height() - HEIGHT_BIAS
            collided = True

        self.x = new_x
        self.y = new_y
        return collided
