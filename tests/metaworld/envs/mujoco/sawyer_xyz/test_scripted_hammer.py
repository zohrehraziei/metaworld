import numpy as np

from metaworld.envs.mujoco.env_dict import ALL_ENVIRONMENTS
from tests.metaworld.envs.mujoco.sawyer_xyz.scripted_policies_utils import (
    move_end_effector, close_gripper, task_completed)


def test_scripted_hammer():
    """Test if hammer can nail in the nail in "Hammer-V1"""
    render=True
    env = ALL_ENVIRONMENTS['hammer-v1'](random_init=True)
    for _ in range(1000):
        env.reset_model()
        env.reset()

        # cage hammer
        move_end_effector(env, env.hammer_init_pos + np.array([0., 0., 0.03]))

        # close gripper
        close_gripper(env)

        # lift hammer
        cur_pos = env.get_endeff_pos()
        lift_pos = cur_pos + np.array([0., 0., env.obj_init_pos[2]])
        move_end_effector(env, lift_pos, render=render)

        # align hammer
        cur_pos = env.get_endeff_pos()
        align_pos = cur_pos + np.array([env.obj_init_pos[0]-0.11, 0., 0.])
        move_end_effector(env, align_pos, render=render)

        move_end_effector(env, env.obj_init_pos + np.array([-0.11, 0., 0.03]),
            render=render)

        move_end_effector(env, env.obj_init_pos + np.array([-0.11, 0.3, 0.03]),
            render=render)

        assert task_completed(env)
