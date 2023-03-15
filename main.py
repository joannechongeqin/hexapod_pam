import pygame
from pygame.locals import *
from Yuna import Yuna
import numpy as np
import time
import threading
import pyttsx3

# state global variables
global joystick_added, parking_mode, controller
joystick_added = False
parking_mode = True # parking mode applies a small coefficient so that the robot can do more subtle movements to finetune its position and orientation, default is on 
# initialisation
pygame.init()
pygame.joystick.init()
voice = pyttsx3.init()# voice module
voice.setProperty('rate', 100)
yuna = Yuna(real_robot_control=0, pybullet_on=1)

def process_joystick(controller): # read step parameters from the controller joysticks
    x1 = - 0.2 * controller.get_axis(1)
    y1 = - 0.2 * controller.get_axis(0)
    x2 = - 20 * controller.get_axis(3)
    step_len = np.sqrt(x1 * x1 + y1 * y1)
    course = np.rad2deg(np.arctan2(y1, x1))
    rotation = x2
    return step_len, course, rotation

def read_controller_thread():
    global joystick_added, parking_mode, controller
    print('Waiting for controller...')
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
                    voice.say('Parking mode, ' + ('on' if parking_mode else 'off'))
                    voice.runAndWait()
                if event.button == 2: #button X
                    print('+++++YUNA DISCONNECTED+++++')
                    voice.say('Yuna disconnected')
                    voice.runAndWait()
                    time.sleep(2)
                    yuna.disconnect()
        time.sleep(0.1)# sleep to prevent excessive CPU usage of this daemon thread
try:
    read_controller = threading.Thread(target=read_controller_thread)
    read_controller.daemon = True
    read_controller.start()
except:
    print('Error: Unable to start thread')

# main thread
while True:
    if joystick_added:
        speed_coef = 0.2 if parking_mode else 1
        step_len, course, rotation = process_joystick(controller)
        if np.abs(step_len) < 0.01 and np.abs(rotation) < 1:
            yuna.stop()
        else:
            yuna.step(speed_coef*step_len, course, speed_coef*rotation)