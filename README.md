Run main.py to control the robot using a game controller
Run joystick_test.py to test the game controller and obtain button index
Run motion_test.py to watch yuna controlled by a series of commands

Yuna configuration
   ^^ Front ^^ 
    1 ----- 2       +x
        |           ^
    3 ----- 4       |  <-╮+yaw 
        |     +y <--o    
    5 ----- 6       +z     

Three functions for yuna maneuvering:
    walk()
        parameters:
            stride : float
                Stride length (in metre) that robot will move every step, but the robot will not always move the exact distance, here are some exceptions:
                    1. When robot is in initial position, the robot will move stride/2 to start walking
                    2. When stride==0, and robot is in the process of walking, the robot will step another stride/2 to recover to initial position
                The value is clipped in [-0.3, 0.3] to avoid collision, but the positive value is encouraged for a more intuitive control
            angle : float
                Moving direction (in degree) the robot will move to
            step : int
                Steps robot will move, the step for starting is counted. If the robot directly changes the maneuvering mode (e.g. turning -> walking) 
                without executing the stop() funtion, the robot will automatically insert the stop() in between and this step is not counted

    turn()
        parameters:
            deg_per_step : float
                Angle (in degree) that robot body will rotate every step, positive for counter-clock wise and negetive for clock wise, but the robot will not always rotate the exact angle, here are some exceptions:
                    1. When robot is in initial position, the robot will rotate deg_per_step/2 to start turning
                    2. When deg_per_step==0, and the robot is in the process of turning, the robot will turn another angle to recover to initial position
                    The value is clipped in [-30, 30] to avoid collision
            step : int
                Steps of turning robot will rotate, the step for starting is counted. If the robot directly changes the maneuvering mode (e.g. walking -> turning) 
                without executing the stop() funtion, the robot will automatically insert the stop() in between and this step is not counted
    
    stop()
        By executing this, the robot will do an extra step to recover to the initial standing pose, if robot is already standing still, the robot will do nothing