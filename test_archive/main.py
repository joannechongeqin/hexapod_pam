import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pygame
from pygame.locals import *
from Yuna import Yuna
import numpy as np
import time
import threading

# state global variables
global joystick_added, parking_mode, controller
joystick_added = False
parking_mode = True # parking mode applies a small coefficient so that the robot can do more subtle movements to finetune its position and orientation, default is on 
# initialisation
pygame.init()
pygame.joystick.init()
voice_on = False
if voice_on:
    import pyttsx3
    voice = pyttsx3.init()# voice module
    voice.setProperty('rate', 100)
yuna = Yuna(real_robot_control=0, pybullet_on=1)
# yuna.smoothing = False
#================================================================================================
def process_joystick(controller):
    '''
    Read step parameters from the controller joysticks
    :param controller: the controller object
    :return step_len: step length
    :return course: course angle
    :return rotation: rotation angle
    '''
    x1 = - 0.2 * controller.get_axis(1)
    y1 = - 0.2 * controller.get_axis(0)
    x2 = - 20 * controller.get_axis(3)
    step_len = np.sqrt(x1 * x1 + y1 * y1)
    course = np.rad2deg(np.arctan2(y1, x1))
    rotation = x2
    return step_len, course, rotation

def confirmation_feedback(text):
    '''
    Audiation the command and print it on the terminal for confirmation
    :param text: the command to be audiated and printed
    :return: None
    '''
    if voice_on:
        print(text)
        voice.say(text)
        if voice._inLoop:
            voice.endLoop()
        voice.startLoop(False)
        voice.iterate()
        time.sleep(2)
        voice.endLoop()
        # p.s. the voice module is not used in a typical way, but this is a way that works without bugs for now

command_frequency = 15 # Hz
def read_controller_thread():
    '''
    A second loop thread besides the main thread that reads the controller and sends commands to the robot at a constant frequency
    :return: None
    '''
    global joystick_added, parking_mode, controller
    print('Waiting for controller...')
    while True:
        t_start = time.perf_counter()
        for event in pygame.event.get():
            if event.type == JOYDEVICEADDED:
                controller = pygame.joystick.Joystick(0)
                joystick_added = True
                confirmation_feedback('CONTROLLER CONNECTED')
            if event.type == JOYBUTTONDOWN: # long press for about 1s or press when all movements are done
                if event.button == 0: #button A
                    parking_mode = not parking_mode
                    confirmation_feedback('PARKING MODE ' + ('ON' if parking_mode else 'OFF'))
                if event.button == 2: #button X
                    yuna.stop()
                    confirmation_feedback('YUNA DISCONNECTED')
                    yuna.disconnect()
        if joystick_added:
            speed_coef = 0.2 if parking_mode else 1
            step_len, course, rotation = process_joystick(controller)
            step_len = speed_coef * step_len
            rotation = speed_coef * rotation
            step_len = step_len if np.abs(step_len) > 0.01 else 0
            rotation = rotation if np.abs(rotation) > 1 else 0
            yuna.get_step_params(step_len, course, rotation)
        t_end = time.perf_counter()
        t = t_end - t_start
        time.sleep(max(1 / command_frequency - t, 0)) # sleep to maintain the desired command frequency and avoid excessive CPU usage of this daemon thread
#================================================================================================
# read controller thread
try:
    read_controller = threading.Thread(target=read_controller_thread)
    read_controller.daemon = True
    read_controller.start()
except:
    print('Error: Unable to start thread, please restart the program')
    raise SystemExit
#================================================================================================
# main thread
step_made = False
while True:
    if not step_made:
        time.sleep(0.1)
    step_made = yuna.step()