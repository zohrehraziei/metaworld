"""Implements functions for creating scripted policies for task environments.

examples of functions are move_end_effector, close_gripper, open_gripper, etc.
"""
import numpy as np


def move_end_effector(env, goal_location, num_sim_steps=100, render=False):
    """Move the position of the end effector to goal_location.

    Args:
        env (Metaworld.envs.mujoco.SawyerXYZEnv): The mujoco-py sawyer
            environment.
        goal_location (numpy.ndarray): The location to reposition the end
            effector to. Shape is `(3,)`
        num_sim_steps(int): The number of steps taken within the mujoco
            simulator.
        render(bool): Whether or not the environment should be render after
            the end effector has been moved.

    """
    env.data.set_mocap_pos('mocap', goal_location)
    env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
    for _ in range(num_sim_steps):
        env.sim.step()
        if render:
            env.render()

def close_gripper(env, num_sim_steps=100, render=False):
    """Close the Sawyer gripper.

    Args:
        env (Metaworld.envs.mujoco.SawyerXYZEnv): The mujoco-py sawyer
            environment.
        num_sim_steps(int): The number of steps taken within the mujoco
            simulator.
        render(bool): Whether or not the environment should be render after
            the end effector has been moved.

    """
    for _ in range(num_sim_steps):
        env.sim.data.ctrl[:] = [1,-1]
        env.sim.step()
        if render:
            env.render()

def open_gripper(env, num_sim_steps=100, render=False):
    """Open the sawyer gripper.

    Args:
        env (Metaworld.envs.mujoco.SawyerXYZEnv): The mujoco-py sawyer
            environment.
        num_sim_steps(int): The number of steps taken within the mujoco
            simulator.
        render(bool): Whether or not the environment should be render after
            the end effector has been moved.

    """
    for _ in range(num_sim_steps):
        env.sim.data.ctrl[:] = [-1, 1]
        env.sim.step()
        if render:
            env.render()

def task_completed(env):
    """Verify that the environment task has been carried out succesfully.

    Args:
        env (Metaworld.envs.mujoco.SawyerXYZEnv): The mujoco-py sawyer
            environment.

    """
    _, _, _, info = env.step(env.action_space.sample())
    return info['success']
