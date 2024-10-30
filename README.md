# Racecar
Autonomous racing

# Project Overview

This project is a learning initiative based on the F1tenth environment, aimed at enhancing the understanding and application of autonomous driving technologies through hands-on practice. The project is divided into three stages, each focusing on optimizing the performance of an F1tenth race car in a simulated environment.

## Stage One: Trajectory Tracking
In this stage, we implement a tracking algorithm using Lidar technology to follow the centerline of the Monza track, represented as a series of points. This practice aims to delve into the fundamental principles and technologies of trajectory tracking.

## Stage Two: Path Planning and Control
The goal of the second stage is to achieve faster racing speeds. This stage includes two main parts:
- **Path Planning**: Develop a path planning algorithm that maximizes the total curvature radius of the path, thereby planning the fastest possible driving speed.
- **Path Control**: Implement and compare three different path tracking methods: Pure Pursuit, Stanley, and Model Predictive Control (MPC).

## Stage Three: SLAM Implementation and Map Comparison
In the final stage of the project, we implement Simultaneous Localization and Mapping (SLAM) to ensure the vehicle travels along a chosen circuit without collision. Additionally, the accuracy and effectiveness of the constructed map are evaluated by comparing it with ground truth data.
[Click here to watch the demo video](https://github.com/Barry-coy-LA/Racecar/blob/main/Phase%203/output_video.mp4)

## Learning Objectives
Through this project, participants will be able to:
- Master the technology of trajectory tracking using Lidar data.
- Understand and implement efficient path planning strategies.
- Learn and apply various path control technologies, comparing their advantages and disadvantages.
- Master SLAM technology and apply it effectively in dynamic environments.
- Gain a deep understanding of the operational mechanisms and control strategies of autonomous vehicles.

## Technologies Used
The project is developed in Python, relying on the F1tenth simulation environment for testing and validation.
