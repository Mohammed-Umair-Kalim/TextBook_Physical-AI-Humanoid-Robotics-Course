---
sidebar_position: 1
---

# Principles of Bipedal Locomotion

## Overview

Bipedal locomotion, or walking on two legs, is one of the defining characteristics of a humanoid robot. It is also one of the most difficult challenges in robotics. This section introduces two of the most important concepts in bipedal locomotion: the Zero Moment Point (ZMP) and the Linear Inverted Pendulum Model (LIPM).

## Detailed Explanation

### The Zero Moment Point (ZMP)

The Zero Moment Point (ZMP) is a concept that was first introduced by Miomir Vukobratović in the early 1970s. It is defined as the point on the ground where the net moment of the inertial forces and the gravity forces has no component along the horizontal axes. In simpler terms, the ZMP is the point on the ground where the total tipping moment acting on the robot is zero.

For the robot to be stable, the ZMP must always remain within the **support polygon**. The support polygon is the area on the ground that is formed by the convex hull of all the points where the robot is in contact with the ground. For a robot standing on one foot, the support polygon is the area of the foot. For a robot standing on two feet, the support polygon is the area that includes both feet and the space between them.

By controlling the ZMP, it is possible to control the stability of the robot. Most modern bipedal robots use some form of ZMP-based control to generate their walking patterns. The basic idea is to first plan a desired trajectory for the ZMP, and then use a control system to move the robot's body in such a way that the actual ZMP tracks the desired ZMP.

### The Linear Inverted Pendulum Model (LIPM)

The Linear Inverted Pendulum Model (LIPM) is a simplified model of a bipedal robot that is often used for walking pattern generation. In this model, the entire mass of the robot is assumed to be concentrated at a single point (the center of mass), and the legs are assumed to be massless. The robot is modeled as a point mass on top of a rigid rod, which is free to pivot about a point on the ground.

The dynamics of the LIPM are much simpler than the dynamics of a full humanoid robot, which makes it possible to generate walking patterns in real-time. The basic idea is to control the position of the center of mass in such a way that the ZMP remains within the support polygon.

By combining the LIPM with ZMP-based control, it is possible to generate smooth and stable walking patterns for a humanoid robot. This approach has been used successfully in many famous humanoid robots, including Honda's ASIMO and Boston Dynamics' Atlas.

---

## Hands-on Exercise: Simulating a Linear Inverted Pendulum

This exercise will guide you through writing a Python script to simulate the motion of a Linear Inverted Pendulum (LIPM).

### The Scenario
You have a point mass `m` at a height `h`. The position of the point mass is `(x, y, h)`. The position of the pivot point on the ground is `(px, py)`. The dynamics of the LIPM are given by the following equations:

`x_ddot = (g / h) * (x - px)`
`y_ddot = (g / h) * (y - py)`

where `g` is the acceleration due to gravity.

Your task is to write a function that takes the current state of the pendulum `(x, y, x_dot, y_dot)` and the position of the pivot `(px, py)` as input, and returns the accelerations `(x_ddot, y_ddot)`. You will then use this function to simulate the motion of the pendulum over time.

### The Code
This Python script simulates the LIPM.

```python
import matplotlib.pyplot as plt

def lipm_dynamics(state, pivot, g, h):
    """
    Calculates the accelerations of the LIPM.

    Args:
        state: A tuple (x, y, x_dot, y_dot) representing the current state of the pendulum.
        pivot: A tuple (px, py) representing the position of the pivot.
        g: The acceleration due to gravity.
        h: The height of the center of mass.

    Returns:
        A tuple (x_ddot, y_ddot) representing the accelerations of the pendulum.
    """
    x, y, _, _ = state
    px, py = pivot
    x_ddot = (g / h) * (x - px)
    y_ddot = (g / h) * (y - py)
    return (x_ddot, y_ddot)

def simulate_lipm(initial_state, pivot, g, h, dt, num_steps):
    """
    Simulates the motion of the LIPM over time.

    Args:
        initial_state: The initial state of the pendulum.
        pivot: The position of the pivot.
        g: The acceleration due to gravity.
        h: The height of the center of mass.
        dt: The time step for the simulation.
        num_steps: The number of steps to simulate.

    Returns:
        A list of states representing the trajectory of the pendulum.
    """
    states = [initial_state]
    for _ in range(num_steps):
        current_state = states[-1]
        x, y, x_dot, y_dot = current_state
        x_ddot, y_ddot = lipm_dynamics(current_state, pivot, g, h)
        x_dot += x_ddot * dt
        y_dot += y_ddot * dt
        x += x_dot * dt
        y += y_dot * dt
        states.append((x, y, x_dot, y_dot))
    return states

if __name__ == '__main__':
    # Simulation parameters
    initial_state = (0.0, 0.0, 0.5, 0.0)  # (x, y, x_dot, y_dot)
    pivot = (0.1, 0.0)
    g = 9.81
    h = 1.0
    dt = 0.01
    num_steps = 200

    # Simulate the LIPM
    trajectory = simulate_lipm(initial_state, pivot, g, h, dt, num_steps)

    # Plot the results
    x_traj = [state[0] for state in trajectory]
    y_traj = [state[1] for state in trajectory]
    plt.plot(x_traj, y_traj)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Trajectory of a Linear Inverted Pendulum")
    plt.grid(True)
    plt.axis('equal')
    plt.show()
```

### How it Works
1.  The `lipm_dynamics` function implements the equations of motion for the LIPM.
2.  The `simulate_lipm` function uses Euler integration to simulate the motion of the pendulum over time.
3.  The main part of the script sets up the simulation parameters, runs the simulation, and plots the results.

### Experiment
*   Run the script and observe the trajectory of the pendulum.
*   Try changing the initial state of the pendulum and the position of the pivot and see how the trajectory changes.
*   Can you modify the script to simulate walking by changing the position of the pivot at each step?

---

## Case Study: Boston Dynamics' Atlas - A Leap in Dynamic Balancing

**Introduction:**
Boston Dynamics' Atlas is arguably the most advanced humanoid robot in the world. It is capable of a wide range of dynamic behaviors, including running, jumping, and even performing backflips. This case study examines some of the key control strategies that enable Atlas's remarkable agility.

**Beyond ZMP:**
While ZMP-based control is effective for generating stable walking patterns on flat terrain, it is not well-suited for more dynamic behaviors like running and jumping. This is because the ZMP is only defined when the robot is in contact with the ground. During the flight phase of running or jumping, the ZMP is undefined.

To overcome this limitation, Boston Dynamics has developed a more advanced control strategy that is based on a whole-body control approach. This approach considers the full dynamics of the robot and allows for more aggressive and dynamic movements.

**Model Predictive Control (MPC):**
At the heart of Atlas's control system is a form of Model Predictive Control (MPC). MPC is an advanced control strategy that uses a model of the robot to predict its future behavior. The controller then optimizes the robot's control inputs (i.e., the joint torques) over a short time horizon to achieve a desired goal, such as tracking a desired trajectory or maintaining balance.

One of the key advantages of MPC is that it can handle constraints, such as the limits on joint torques and the need to keep the robot's feet on the ground. This makes it well-suited for controlling a complex system like a humanoid robot.

**A Hierarchical Approach:**
Atlas's control system is organized in a hierarchical manner.
*   **High-level planner:** A high-level planner is used to generate a rough plan for the robot's behavior, such as a sequence of footsteps to navigate a complex terrain.
*   **Mid-level controller:** A mid-level controller, based on MPC, is used to refine the plan and generate a smooth and dynamically feasible trajectory for the robot's center of mass.
*   **Low-level controller:** A low-level controller is used to calculate the joint torques required to track the desired trajectory.

**Legacy:**
Atlas has pushed the boundaries of what is possible in humanoid robotics. It has demonstrated that it is possible to build a bipedal robot that is not only stable but also agile and robust. The control strategies developed for Atlas are now being used in a wide range of other robotic systems, and they are likely to play a key role in the development of the next generation of humanoid robots.
