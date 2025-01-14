# README

## Pose Assisted Manipulation (PAM) for a Hexapod Robot
Hexapod robots offer superior stability on uneven terrain compared to quadruped and biped robots. Additionally, they can perform manipulation tasks in challenging environments by utilizing their extra legs. However, the size of the robot often limits the range of its manipulation capabilities. This project aims to design advanced poses for a hexapod robot to extend its manipulators and body, allowing it to access elevated objectives beyond reach of conventional ground-based poses. 

## Yuna Configuration

       ^^ Front ^^   
        1 ----- 2       +x  
            |           ^  
        3 ----- 4       |  <-╮+yaw   
            |     +y <--o    
        5 ----- 6       +z     

## Proposed Overall Pipeline
1. Optimization for the final stable pose of the robot to ensure that its end effectors can accurately reach the target positions required for the manipulation task.
2. Interpolation of intermediate body waypoints if the distance between the robot's initial pose and the optimized final pose is too far to be reached in a single transition.
3. Optimization at each interpolated waypoint for a stable pose based on terrain conditions.
4. Progressive transition to the next pose using pentagonal gait and cubic hermite interpolation, moving the robot through the series of interpolated waypoints. 

## Work in Progress
- [`pam_optimizer.py`](pam_optimizer.py): Contains the main optimization logic using PyTorch.
    - [`static_stability_margin.py`](static_stability_margin.py): Contains functions to calculate the static stability margin of the hexapod to ensure that the optimized poses are stable.
- [`Yuna.py`](Yuna.py): Contains the main functions for controlling the hexapod robot.
- [`YunaEnv.py`](YunaEnv.py): Sets up the simulation environment.
- [`pam_demo.py`](pam_demo.py): Demonstrates how to specify target positions for the legs and use PAM functions. 
