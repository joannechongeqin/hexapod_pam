import pygame
from pygame.locals import *
from Yuna import Yuna
import numpy as np

joystick_added = False
parking_mode = True # parking mode applies a small coefficient so that the robot can do more subtle movements to finetune its position and orientation, default is on 
# initialise
pygame.init()
pygame.joystick.init()
yuna = Yuna(real_robot_control=0, pybullet_on=1)

def read_joystick(controller): # read step parameters from the controller joysticks
    x1 = - 0.2 * controller.get_axis(1)
    y1 = - 0.2 * controller.get_axis(0)
    x2 = - 20 * controller.get_axis(3)
    step_len = np.sqrt(x1 * x1 + y1 * y1)
    course = np.rad2deg(np.arctan2(y1, x1))
    rotation = x2
    return step_len, course, rotation

while True:
    for event in pygame.event.get():
        if event.type == JOYDEVICEADDED:
            controller = pygame.joystick.Joystick(0)
            joystick_added = True
            print('Controller ' + controller.get_name() + ' connected!')
        if event.type == JOYBUTTONDOWN: # long press for about 1s or press when all movements are done
            if event.button == 0: #button A
                parking_mode = not parking_mode
                print('+++++PARKING MODE ' + ('ON' if parking_mode else 'OFF') + '+++++')
            if event.button == 2: #button X
                print('+++++YUNA DISCONNECTED+++++')
                yuna.disconnect()

    if joystick_added: # detect if controller is added
        speed_coef = 0.2 if parking_mode else 1
        step_len, course, rotation = read_joystick(controller)
        if np.abs(step_len) < 0.01 and np.abs(rotation) < 1:
            yuna.stop()
        else:
            yuna.step(speed_coef*step_len, course, speed_coef*rotation)