# Pose Assisted Manipulation (PAM) for a Hexapod Robot

## Introduction 

This project aims to enhance the capabilities of hexapod robots by enabling them to use their legs for both locomotion and manipulation. It involves **the development of a pose optimization algorithm and a motion planning pipeline** for a hexapod robot to perform manipulation tasks at elevated targets, beyond the reach of traditional ground-based configurations.

> *Pose* refers to both:
> (1) the robot’s overall position and orientation in 3D space relative to the world origin, and  
> (2) the position and orientation of each leg (end effector) with respect to the robot's body frame (computed using forward kinematics based on the robot's joint angles).  

Hexapod robots offer exceptional stability on uneven terrain, as they can maintain static stability with three or more legs, allowing them to navigate rough environments with ease. In addition, their redundant legs present opportunities for manipulation, to perform tasks and interact with their surroundings. However, the robot’s size and conventional configuration often limit its manipulation range. Simply walking closer to a target may not always be feasible due to obstacles, terrain conditions, or spatial constraints.

This project was completed as my final year project at the National University of Singapore’s MARMot Lab. A special thanks to my supervisor and the PhD students of the Legged Team for their invaluable support, insightful discussions, and technical guidance throughout the project.


## Environment Representation and Task Specification

For this project, the environment is assumed to be static, rigid, and predetermined. The task involves navigating a terrain with two distinct steps and using leg 2 (the front-right leg) to reach and press a button located at a fixed goal position on the wall.


## Demonstration Video

https://github.com/user-attachments/assets/7211eab6-9b2f-42e7-93e6-b2eb6e9f2099


## Proposed Overall Pipeline

1. **Optimization for the final stable pose** of the robot to ensure that its end effectors can accurately reach the target positions required for the manipulation task.

2. **Interpolation of intermediate body waypoints** if the distance between the robot's initial pose and the optimized final pose is too far to be reached in a single transition.

3. **Optimization at each interpolated waypoint** for a stable pose based on terrain conditions.

4. **Progressive transition to the subsequent pose** using tripod gait and cubic Hermite interpolation, moving the robot through the series of interpolated waypoints.


## Yuna Configuration

This work is done with the [HEBI Robotics `Daisy` 18-DoF (degrees of freedom) hexapod robot](https://robotsguide.com/robots/daisy). 

       ^^ Front ^^   
        1 ----- 2       +x  
            |           ^  
        3 ----- 4       |  <-╮+yaw   
            |     +y <--o    
        5 ----- 6       +z     
        

## Code Structure
| File | Description |
|------|-------------|
| [`pam_optimizer.py`](pam_optimizer.py) | Main pose optimization logic using PyTorch. |
| [`static_stability_margin.py`](static_stability_margin.py) | Functions to calculate the static stability margin of the hexapod to ensure stability of optimized poses. |
| [`Yuna.py`](Yuna.py) | Main functions for controlling the hexapod robot. |
| [`Yuna_TrajPlanner.py`](Yuna_TrajPlanner.py) | Computes pose keyframes and trajectories |
| [`YunaEnv.py`](YunaEnv.py) | Sets up the simulation environment. |
| [`zed_camera.py`](zed_camera.py) | Integrates the ZED 2i stereo camera for positional tracking as odometry feedback in real world testing. |
| [`pam_demo.py`](pam_demo.py) | Demonstrates how to specify target positions for the legs and use PAM functions. |
