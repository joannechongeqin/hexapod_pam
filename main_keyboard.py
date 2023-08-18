'''
Keyboard control for Yuna, the major keys used to control the robot are:
    W: move forward
    A: move left
    S: move backward
    D: move right
    Q: rotate counter-clockwise
    E: rotate clockwise
    P: toggle parking mode
    ESC: exit
'''

from Yuna import Yuna
import numpy as np
import time
from pynput import keyboard
import pyttsx3

# initialisation
yuna = Yuna(real_robot_control=0, pybullet_on=1)

# define global variables
global left_pressed, right_pressed, up_pressed, down_pressed, cw_pressed, ccw_pressed
left_pressed = False
right_pressed = False
up_pressed = False
down_pressed = False
cw_pressed = False
ccw_pressed = False
global parking_mode
parking_mode = True

def process_keyboard():
    global left_pressed, right_pressed, up_pressed, down_pressed, cw_pressed, ccw_pressed
    global parking_mode
    speed_coef = 0.2 if parking_mode else 1.0

    x = int(up_pressed) - int(down_pressed)
    y = int(left_pressed) - int(right_pressed)
    step_len = speed_coef * 0.2 * np.sqrt(x * x + y * y)
    course = np.rad2deg(np.arctan2(y, x))
    rotation = speed_coef * 20 * (int(ccw_pressed) - int(cw_pressed))
    yuna.get_step_params(step_len, course, rotation)

# voice feedback feature
voice_on = False
if voice_on:
    voice = pyttsx3.init()# voice module
    voice.setProperty('rate', 100)
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
#================================================================================================
def on_press(key):
    global left_pressed, right_pressed, up_pressed, down_pressed, cw_pressed, ccw_pressed
    try:
        if key.char == 'w' or key.char == 'W':
            up_pressed = True
        if key.char == 'a' or key.char == 'A':
            left_pressed = True
        if key.char == 's' or key.char == 'S':
            down_pressed = True
        if key.char == 'd' or key.char == 'D':
            right_pressed = True
        if key.char == 'q' or key.char == 'Q':
            ccw_pressed = True
        if key.char == 'e' or key.char == 'E':
            cw_pressed = True
        process_keyboard()
    except AttributeError:
        pass

def on_release(key):
    global left_pressed, right_pressed, up_pressed, down_pressed, cw_pressed, ccw_pressed
    global parking_mode
    try:
        if key.char == 'w' or key.char == 'W':
            up_pressed = False
        if key.char == 'a' or key.char == 'A':
            left_pressed = False
        if key.char == 's' or key.char == 'S':
            down_pressed = False
        if key.char == 'd' or key.char == 'D':
            right_pressed = False
        if key.char == 'q' or key.char == 'Q':
            ccw_pressed = False
        if key.char == 'e' or key.char == 'E':
            cw_pressed = False
        if key.char == 'p' or key.char == 'P':
            parking_mode = not parking_mode
            confirmation_feedback('PARKING MODE ' + ('ON' if parking_mode else 'OFF'))
            
        process_keyboard()

    except AttributeError:
        # press ESC to exit
        if key == keyboard.Key.esc:
            yuna.stop()
            confirmation_feedback('YUNA DISCONNECTED')
            yuna.disconnect()
            return False
#================================================================================================
# keyboard listener thread
try:
    listener = keyboard.Listener(on_press=on_press, on_release=on_release, suppress=True)
    listener.start()
except:
    print('Keyboard listener failed to start')
    raise SystemExit
#================================================================================================
# main thread
step_made = False
while True:
    if not step_made:
        time.sleep(0.1)
    step_made = yuna.step()