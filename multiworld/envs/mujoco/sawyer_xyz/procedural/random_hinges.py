from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv
from multiworld.envs.mujoco.sawyer_xyz.procedural import model_builder
from gym.envs.mujoco import mujoco_env
from gym.spaces import  Dict, Box

import numpy as np
import random

class DoorSide():
    def __init__(self, idx, sign, hinge_sign):
        """
        idx : must be 0, or 1
        sign: must be -1 or 1
        hinge_idx: must be 
        """ 
        self.idx = idx
        self.sign = sign
        self.hinge_idx = (idx + 1) % 2
        self.hinge_sign = hinge_sign

def hinge(door_side):
    mjcmodel = model_builder.MJCModel(name='hinge')
    # mjcmodel.root.compiler(inertiafromgeom="auto",
    #     angle="radian",
    #     coordinate="local", 
    #     eulerseq="XYZ")

    # mjcmodel.root.size(njmax=6000, nconmax=6000)
    # mjcmodel.root.option(timestep="0.005", gravity="0 0 -9.81", iterations="50", integrator="Euler")
    # default = mjcmodel.root.default()
    # default.joint(limited="false", damping="1")
    # default.geom(contype="1", conaffinity="1", condim="3", friction=".5 .1 .1", density="1000", margin="0.002")

    include = mjcmodel.root.include(file="shared_config.xml")

    worldbody = mjcmodel.root.worldbody()

    actuator = mjcmodel.root.actuator()
    actuator.position(ctrllimited="true", ctrlrange="-1 1", joint="r_close", kp="400", user="1")
    actuator.position(ctrllimited="true", ctrlrange="-1 1", joint="l_close", kp="400", user="1")


    # worldbody.camera(name="maincam", mode="fixed", fovy="32", euler="0.7 0 0", pos="0 -1.1 1.3")
    # worldbody.camera(name="leftcam", mode="fixed", fovy="32", euler="0.7 0 -1.57", pos="-1.1 0 1.3")
    # worldbody.camera(name="overheadcam", mode= "fixed", pos="0. 0. 1.5", euler="0.0 0.0 0.0")


    include_robot = worldbody.include(file="sawyer_xyz_base_no_table.xml")

    angle = random.random() * 2 * np.pi
    # angle = 0

    box1_pos = [0, 0.6, .07]
    box1_size = [.07, .07, .07]
    box = worldbody.body(name="box", pos=box1_pos, axisangle="0 0 1 {}".format(angle))
    box.inertial(pos="0 0 0", mass="1", diaginertia="1 1 1")
    box.geom(pos="0 0 0", name="base_obj", type="box", size="{} {} {}".format(box1_size[0], box1_size[1], box1_size[2]), rgba="0.3 1 0.3 1", contype="1", conaffinity="1")
    

    box2_size = [.09, .09, .09]
    idx = door_side.idx
    box2_size[door_side.idx] = .02
    # box2_pos = box1_pos[:]
    box2_pos = [0, 0, 0]

    sign = door_side.sign
    box2_pos[idx] = box1_pos[idx] + sign*(.02 + box1_size[idx] + box2_size[idx])

    box2 = worldbody.body(name="box2", pos=box1_pos, axisangle="0 0 1 {}".format(angle))
    box2.inertial(pos="0 0 0", mass="1", diaginertia="1e-6 1e-6 1e-6")
    box2.geom(pos=box2_pos, name="target_obj", type="box", size="{} {} {}".format(box2_size[0], box2_size[1], box2_size[2]), rgba="0.3 .3 1 1", contype="1", conaffinity="1")
    joint_pos = [0, 0, 0]
    joint_pos[door_side.hinge_idx] = door_side.hinge_sign * box1_size[2]
    box2.joint(pos="{} {} {}".format(joint_pos[0], joint_pos[1], joint_pos[2]), name="hinge", range="-3 3", axis="0 0 1", damping=".001", armature="0")

    return mjcmodel

class HingeEnv(SawyerXYZEnv):

    def __init__(self,
        door_side,
        rotMode='rotz',
        hand_low=(-0.5, 0.40, 0.05),
        hand_high=(0.5, 1, 0.5),
        # Todo: maybe make these types safe.
        **kwargs):
        self.rotMode = rotMode
        self.hand_init_pos = np.array([0, 0.6, 0.2])
        self._state_goal = 0
        self.max_path_length = 1000

        model = hinge(door_side)
        with model.asfile() as f:
            with open("/Users/deirdrequillen/metaworld/multiworld/envs/assets/sawyer_xyz/random_hinge.xml", "w") as new:
                new.write(f.read())

        SawyerXYZEnv.__init__(
            self,
            frame_skip=5,
            action_scale=1./100,
            hand_low=hand_low,
            hand_high=hand_high,
            model_name="/Users/deirdrequillen/metaworld/multiworld/envs/assets/sawyer_xyz/random_hinge.xml",
            **kwargs
        )

        # with open("random_hinge.xml", "w") as new:
        #     new.write(model.__str__())

        # with model.asfile() as f:
            # mujoco_env.MujocoEnv.__init__(self, f.name, 5)
        # mujoco_env.MujocoEnv.__init__(self, "/Users/deirdrequillen/metaworld/multiworld/envs/assets/sawyer_xyz/random_hinge.xml", 5)
        
        if rotMode == 'fixed':
            self.action_space = Box(
                np.array([-1, -1, -1, -1]),
                np.array([1, 1, 1, 1]),
            )
        elif rotMode == 'rotz':
            self.action_rot_scale = 1./50
            self.action_space = Box(
                np.array([-1, -1, -1, -np.pi, -1]),
                np.array([1, 1, 1, np.pi, 1]),
            )
        elif rotMode == 'quat':
            self.action_space = Box(
                np.array([-1, -1, -1, 0, -1, -1, -1, -1]),
                np.array([1, 1, 1, 2*np.pi, 1, 1, 1, 1]),
            )
        else:
            self.action_space = Box(
                np.array([-1, -1, -1, -np.pi/2, -np.pi/2, 0, -1]),
                np.array([1, 1, 1, np.pi/2, np.pi/2, np.pi*2, 1]),
            )

        self.observation_space = Box(
            np.hstack((hand_low, 0)),
            np.hstack((hand_high, 2*np.pi)),
        )

    def step(self, action):
        if self.rotMode == 'euler':
            action_ = np.zeros(7)
            action_[:3] = action[:3]
            action_[3:] = euler2quat(action[3:6])
            self.set_xyz_action_rot(action_)
        elif self.rotMode == 'fixed':
            self.set_xyz_action(action[:3])
        elif self.rotMode == 'rotz':
            self.set_xyz_action_rotz(action[:4])
        else:
            self.set_xyz_action_rot(action[:7])
        self.do_simulation([action[-1], -action[-1]])
        # The marker seems to get reset every time you do a simulation
        obs = self._get_obs()
        # obs_dict = self._get_obs_dict()
        reward = self.compute_reward(action, obs)
        self.curr_path_length +=1
        #info = self._get_info()
        if self.curr_path_length == self.max_path_length:
            done = True
        else:
            done = False
        return obs, reward, done, None
   
    def _get_obs(self):
        hand = self.get_endeff_pos()
        objPos =  self.data.get_geom_xpos('target_obj')
        flat_obs = np.concatenate((hand, objPos))
        # print(self.get_angle())
        return np.hstack([
            flat_obs,
            self._state_goal
        ])

    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def get_angle(self):
        return np.array([self.data.get_joint_qpos('hinge')])

    def reset_model(self):
        self.curr_path_length = 0
        self._reset_hand()
        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1,1], self.frame_skip)
            #self.do_simulation(None, self.frame_skip)
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False

    def compute_rewards(self, actions, obsBatch):
        #Required by HER-TD3
        assert isinstance(obsBatch, dict) == True
        obsList = obsBatch['state_observation']
        rewards = [self.compute_reward(action, obs)[0] for  action, obs in zip(actions, obsList)]
        return np.array(rewards)

    def compute_reward(self, actions, obs):
        actual_angle = self.get_angle()
        return abs(actual_angle)
        # return np.linalg.norm(actual_angle - goal_angle)

if __name__ == '__main__':
    door_side = DoorSide(0, -1, -1)
    env = HingeEnv(door_side, rotMode = 'rotz')
    for _ in range(1000):
        env.reset()
        env.render()
        env.step(np.array([0, 0, 0, 0, 1]))
        # time.sleep(0.05)


