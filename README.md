# README

## Pose Assisted Manipulation (PAM) of Hexapod
To design advanced poses for a hexapod robot to extend its manipulators and body, allowing it to access elevated objectives beyond reach of conventional ground-based poses

## Yuna Configuration

       ^^ Front ^^   
        1 ----- 2       +x  
            |           ^  
        3 ----- 4       |  <-╮+yaw   
            |     +y <--o    
        5 ----- 6       +z     

## Work in Progress
- [`pam_optimizer.py`](pam_optimizer.py): Contains the main optimization logic using PyTorch.
    - [`static_stability_margin.py`](static_stability_margin.py): Contains functions to calculate the static stability margin of the hexapod to ensure that the optimized poses are stable.
- [`Yuna.py`](Yuna.py): Contains the main functions for controlling the hexapod robot.
- [`YunaEnv.py`](YunaEnv.py): Sets up the simulation environment.
- - [`pam_demo.py`](pam_demo.py): Demonstrates how to specify target positions for the legs and use PAM functions. 
