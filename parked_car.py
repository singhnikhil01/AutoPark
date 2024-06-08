import pygame

'''
function to initialize the cars in the environment.

takes in 4 params: screen of the pygame, the x and y coordinates of the point where the image of the car is to be added
and the angle at which the car is to be placed.
'''
class ParkedCar:
    def __init__(self, screen, x, y, angle=0):
        self.screen = screen
        self.width = 50
        self.height = 90
        self.car_image = pygame.image.load("assets/car_other.png")
        self.car_image = pygame.transform.scale(self.car_image, (self.width, self.height))
        self.x = x
        self.y = y
        self.angle = angle
        self.update_rect()

    def draw(self):
        rotated_image = pygame.transform.rotate(self.car_image, self.angle)
        self.screen.blit(rotated_image, self.rect)

    def update_rect(self):
        rotated_image = pygame.transform.rotate(self.car_image, self.angle)
        self.rect = rotated_image.get_rect(center=(self.x, self.y))
