import abc
import numpy as np
import mujoco_py

from multiworld.core.serializable import Serializable
from multiworld.envs.mujoco.mujoco_env import MujocoEnv
from pyquaternion import Quaternion
from multiworld.envs.env_util import quat_to_zangle, zangle_to_quat
from multiworld.envs.mujoco.utils.inverse_kinematics import qpos_from_site_pose

import copy


class SawyerIKBase(MujocoEnv, Serializable, metaclass=abc.ABCMeta):
    """
    Provides some commonly-shared functions for Sawyer Mujoco envs that use
    IK for XYZ control.
    """

    def __init__(self, model_name, frame_skip=50):
        MujocoEnv.__init__(self, model_name, frame_skip=frame_skip)

    def get_endeff_pos(self):
        return self.sim.data.get_site_xpos('endEffector').copy()

    def get_gripper_pos(self):
        return np.array([self.data.qpos[7]])

    def get_env_state(self):
        joint_state = self.sim.get_state()
        return copy.deepcopy(state)

    def set_env_state(self, state):
        joint_state = state
        self.sim.set_state(joint_state)
        self.sim.forward()

    def __getstate__(self):
        state = super().__getstate__()
        return {**state, 'env_state': self.get_env_state()}

    def __setstate__(self, state):
        super().__setstate__(state)
        self.set_env_state(state['env_state'])

class SawyerXYZIKEnv(SawyerIKBase, metaclass=abc.ABCMeta):
    def __init__(
            self,
            *args,
            hand_low=(-0.2, 0.55, 0.05),
            hand_high=(0.2, 0.75, 0.3),
            action_scale=2./100,
            action_rot_scale=1.,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.action_scale = action_scale
        self.action_rot_scale = action_rot_scale
        self.hand_low = np.array(hand_low)
        self.hand_high = np.array(hand_high)
        self.joint_names = self.sim.model.joint_names[:7]

    def set_xyz_action(self, action):
        action = np.clip(action, -1, 1)
        pos_delta = action * self.action_scale
        new_pos = self.get_endeff_pos() + pos_delta[None]
        new_pos = np.clip(
            new_pos,
            self.hand_low,
            self.hand_high,
        )
        qpos_from_site_pose(self,
                            'endEffector',
                            target_pos=new_pos,
                            target_quat=np.array([1, 0, 1, 0]),
                            joint_names=self.joint_names)

    def set_xyz_action_rot(self, action):
        action[:3] = np.clip(action[:3], -1, 1)
        pos_delta = action[:3] * self.action_scale
        new_pos = self.get_endeff_pos() + pos_delta[None]
        new_pos = np.clip(
            new_pos,
            self.hand_low,
            self.hand_high,
        )
        rot_axis = action[4:] / np.linalg.norm(action[4:])
        action[3] = action[3] * self.action_rot_scale
        # replace this with learned rotation
        new_quat = (Quaternion(axis=[0,1,0], angle=(np.pi)) * Quaternion(axis=list(rot_axis), angle=action[3])).elements
        qpos_from_site_pose(self,
                            'endEffector',
                            target_pos=new_pos,
                            target_quat=new_quat,
                            joint_names=self.joint_names)

    def set_xyz_action_rotz(self, action):
        action[:3] = np.clip(action[:3], -1, 1)
        pos_delta = action[:3] * self.action_scale
        new_pos = self.get_endeff_pos() + pos_delta[None]
        new_pos = np.clip(
            new_pos,
            self.hand_low,
            self.hand_high,
        )
        zangle_delta = action[3] * self.action_rot_scale
        new_zangle = quat_to_zangle(self.sim.model.site_quat[self.sim.model.site_name2id('endEffector')][0]) + zangle_delta
        new_zangle = np.clip(
            new_zangle,
            -3.0,
            3.0,
        )
        if new_zangle < 0:
            new_zangle += 2 * np.pi
        qpos_from_site_pose(self,
                            'endEffector',
                            target_pos=new_pos,
                            target_quat=zangle_to_quat(new_zangle),
                            joint_names=self.joint_names)

    def set_xy_action(self, xy_action, fixed_z):
        delta_z = fixed_z - self.get_endeff_pos()[2]
        xyz_action = np.hstack((xy_action, delta_z))
        self.set_xyz_action(xyz_action)
