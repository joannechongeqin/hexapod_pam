# README
## How to use

Run `main.py` to control the robot using a game controller  
Run `joystick_test.py` to test the game controller and obtain button index  
Run `Yuna.py` to watch yuna controlled by a series of commands
Run `Yuna_TrajPlanner.py` to visualise thetrajectory of one foot

## Yuna configuration

       ^^ Front ^^   
        1 ----- 2       +x  
            |           ^  
        3 ----- 4       |  <-╮+yaw   
            |     +y <--o    
        5 ----- 6       +z     

## Functions for yuna maneuvering
- `step()`
    - parameters:
        - `step_len : float`  
        Stride length (in metre) that robot will move every step, but the robot will not always move the exact distance, it also depends on the previous commands, here are some exceptions:
            1. When robot is in initial position, the robot will move `stride/2` to start walking
            2. When `stride==0`, and robot is in the process of walking, the robot will step another `stride/2` to recover to initial position
        The value is clipped in `[-0.2, 0.2]` to avoid collision, but the positive value is encouraged for a more intuitive control  

        - `course : float`  
        Moving direction (in degree) the robot will move to  

        - `rotation : float`
        Angle (in degree) that robot body will rotate every step, positive for counter-clock wise and negetive for clock wise, but the robot will not always rotate the exact angle, it also depends on the previous commands, here are some exceptions:  
            1. When robot is in initial position, the robot will rotate `rotation/2` to start turning  
            2. When rotation==0, and the robot is in the process of turning, the robot will turn another `rotation/2` angle to recover to initial position
        The value is clipped in [-20, 20] to avoid collision 
            

        - `step : int`  
        Steps robot will move, the step for starting is counted. If the robot directly changes the maneuvering mode (e.g. turning -> walking) without executing the stop() funtion, the robot will automatically insert the stop() in between and this step is not counted    

- `stop()`  
    By executing this, the robot will do an extra step to recover to the initial standing pose, if robot is already standing still, the robot will do nothing  
