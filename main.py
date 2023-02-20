'''
Use a game controller to control Yuna, single click left joystick to switch manouvre mode and single click botton A to turn on/off the parking
mode, the robot's motion is rescaled to a smaller amplitude when parking mode is on.

Strongly recommending run the joystick_test.py before run this python file if you are not using MacOS, to ensure the joystick index mapping correctly
'''
import pygame
from pygame.locals import *
from Yuna import Yuna
import numpy as np
import time

pygame.init()
pygame.joystick.init()
joystick_added = False
walk_mode = True
parking_mode = True

yuna = Yuna(real_robot_control=0)

def read_joystick(controller, walk_mode):
    if walk_mode:
        x = - 0.3 * controller.get_axis(1)
        y = - 0.3 * controller.get_axis(0)
        stride = np.sqrt(x * x + y * y)
        angle = np.rad2deg(np.arctan2(y, x))
        return stride, angle
    else:
        angle = -20 * controller.get_axis(0)
        return angle

while True:
    for event in pygame.event.get():
        if event.type == JOYDEVICEADDED:
            controller = pygame.joystick.Joystick(0)
            joystick_added = True
            print('Controller ' + controller.get_name() + ' connected!')
        if event.type == JOYBUTTONDOWN: # long press for about 1s to toggle mode of manoeuvre
            print(event)
            if event.button == 0: #button A
                parking_mode = not parking_mode
                print('+++++PARKING MODE ' + ('ON' if parking_mode else 'OFF') + '+++++')
            if  event.button == 2: # button X
                print('+++++Bullet server closed+++++')
                yuna.disconnect()
            if event.button == 6:#left joystick
                walk_mode = not walk_mode
                print('+++++THE ROBOT MODE HAS SWITCHED TO ' + ('WALKING' if walk_mode else 'TURNING') + '+++++')

    if joystick_added:
        speed_coef = 0.2 if parking_mode else 1
        if walk_mode:
            stride, angle = read_joystick(controller=controller, walk_mode=walk_mode)
            if stride == 0:
                yuna.stop()
            else:
                yuna.walk(speed_coef * stride, angle)
        else:
            deg_per_step = read_joystick(controller=controller, walk_mode=walk_mode)
            if deg_per_step == 0:
                yuna.stop()
            else:
                yuna.turn(speed_coef * deg_per_step)

