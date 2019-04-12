from collections import OrderedDict
import numpy as np
from gym.spaces import  Dict , Box


from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv

from pyquaternion import Quaternion
from multiworld.envs.mujoco.utils.rotation import euler2quat

class SawyerHandInsert6DOFEnv(SawyerXYZEnv):
    def __init__(
            self,
            hand_low=(-0.5, 0.40, 0.05),
            hand_high=(0.5, 1, 0.5),
            obj_low=(-0.1, 0.5, 0.02),
            obj_high=(0.1, 0.6, 0.02),
            random_init=False,
            tasks = [{'goal': np.array([0., 0.85, 0.001]),  'obj_init_pos':np.array([0, 0.6, 0.02]), 'obj_init_angle': 0.3}], 
            hand_init_pos = (0, 0.6, 0.2),
            liftThresh = 0.04,
            rewMode = 'orig',
            rotMode='fixed',#'fixed',
            multitask=False,
            multitask_num=None,
            task_idx=None,
            **kwargs
    ):
        self.quick_init(locals())
        SawyerXYZEnv.__init__(
            self,
            frame_skip=5,
            action_scale=1./100,
            hand_low=hand_low,
            hand_high=hand_high,
            model_name=self.model_name,
            **kwargs
        )
        if obj_low is None:
            obj_low = self.hand_low


        if obj_high is None:
            obj_high = self.hand_high

        goal_low = self.hand_low

        goal_high = self.hand_high

        

        self.random_init = random_init
        self.liftThresh = liftThresh
        self.max_path_length = 150#200#150
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.rewMode = rewMode
        self.rotMode = rotMode
        self.hand_init_pos = np.array(hand_init_pos)
        self.multitask = multitask
        self.multitask_num = multitask_num
        # self._state_goal_idx = np.zeros(multitask_num)
        self._state_goal_idx = task_idx
        
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
        self.obj_and_goal_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))
        if not multitask:
            self.observation_space = Box(
                    np.hstack((self.hand_low, obj_low, np.zeros(len(tasks)))),
                    np.hstack((self.hand_high, obj_high, np.ones(len(tasks)))),
            )
        else:
            self.observation_space = Box(
                    np.hstack((self.hand_low, obj_low, obj_low, obj_low, goal_low, [0])),
                    np.hstack((self.hand_high, obj_high, obj_high, obj_high, goal_high, [multitask_num - 1])),
            )
        self.reset()


    def get_goal(self):
        return {
            'state_desired_goal': self._state_goal,
    }

    @property
    def model_name(self):     

        return get_asset_full_path('sawyer_xyz/sawyer_table_with_hole_no_puck.xml')
        #return get_asset_full_path('sawyer_xyz/pickPlace_fox.xml')

    def viewer_setup(self):
        # top view
        # self.viewer.cam.trackbodyid = 0
        # self.viewer.cam.lookat[0] = 0
        # self.viewer.cam.lookat[1] = 1.0
        # self.viewer.cam.lookat[2] = 0.5
        # self.viewer.cam.distance = 0.6
        # self.viewer.cam.elevation = -45
        # self.viewer.cam.azimuth = 270
        # self.viewer.cam.trackbodyid = -1
        # side view
        # self.viewer.cam.trackbodyid = 0
        # self.viewer.cam.lookat[0] = 0.2
        # self.viewer.cam.lookat[1] = 0.75
        # self.viewer.cam.lookat[2] = 0.4
        # self.viewer.cam.distance = 0.4
        # self.viewer.cam.elevation = -55
        # self.viewer.cam.azimuth = 180
        # self.viewer.cam.trackbodyid = -1
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] = 0.2
        self.viewer.cam.lookat[1] = 0.5
        self.viewer.cam.lookat[2] = 0.6
        self.viewer.cam.distance = 0.4
        self.viewer.cam.elevation = -55
        self.viewer.cam.azimuth = 135
        self.viewer.cam.trackbodyid = -1
        # robot view
        # self.viewer.cam.trackbodyid = 0
        # self.viewer.cam.lookat[0] = 0
        # self.viewer.cam.lookat[1] = 0.4
        # self.viewer.cam.lookat[2] = 0.4
        # self.viewer.cam.distance = 0.4
        # self.viewer.cam.elevation = -55
        # self.viewer.cam.azimuth = 90
        # self.viewer.cam.trackbodyid = -1

    def step(self, action):
        # self.render()
        # self.set_xyz_action_rot(action[:7])
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
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        obs_dict = self._get_obs_dict()
        reward = self.compute_reward(action, obs_dict, mode = self.rewMode)
        self.curr_path_length +=1
        #info = self._get_info()
        if self.curr_path_length == self.max_path_length:
            done = True
        else:
            done = False
        # return ob, reward, done, { 'reachRew':reachRew, 'reachDist': reachDist, 'pickRew':pickRew, 'placeRew': placeRew, 'epRew' : reward, 'placingDist': placingDist}
        return ob, reward, done, None
   
    def _get_obs(self):
        hand = self.get_endeff_pos()
        # objPos =  self.data.get_geom_xpos('objGeom')
        # if self.multitask:
        #     assert hasattr(self, '_state_goal_idx')
        #     return np.concatenate([
        #             flat_obs,
        #             self._state_goal_idx
        #         ])
        if self.multitask:
            obs = np.hstack([hand, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), self._state_goal_idx])
        else:
            obs = np.concatenate([hand, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)])

        return obs

    def _get_obs_dict(self):
        hand = self.get_endeff_pos()
        # flat_obs = np.concatenate((hand, objPos))
        return dict(
            state_observation=hand,
            state_desired_goal=self._state_goal,
            # state_achieved_goal=objPos,
        )

    def _get_info(self):
        pass
    
    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        self.data.site_xpos[self.model.site_name2id('goal')] = (
            goal[:3]
        )

    # def _set_objCOM_marker(self):
    #     """
    #     This should be use ONLY for visualization. Use self._state_goal for
    #     logging, learning, etc.
    #     """
    #     objPos =  self.data.get_geom_xpos('objGeom')
    #     self.data.site_xpos[self.model.site_name2id('objSite')] = (
    #         objPos
    #     )
    

    # def _set_obj_xyz_quat(self, pos, angle):
    #     quat = Quaternion(axis = [0,0,1], angle = angle).elements
    #     qpos = self.data.qpos.flat.copy()
    #     qvel = self.data.qvel.flat.copy()
    #     qpos[9:12] = pos.copy()
    #     qpos[12:16] = quat.copy()
    #     qvel[9:15] = 0
    #     self.set_state(qpos, qvel)


    # def _set_obj_xyz(self, pos):
    #     qpos = self.data.qpos.flat.copy()
    #     qvel = self.data.qvel.flat.copy()
    #     qpos[9:12] = pos.copy()
    #     qvel[9:15] = 0
    #     self.set_state(qpos, qvel)


    def sample_goals(self, batch_size):
        #Required by HER-TD3
        goals = []
        for i in range(batch_size):
            task = self.tasks[np.random.randint(0, self.num_tasks)]
            goals.append(task['goal'])
        return {
            'state_desired_goal': goals,
        }


    def sample_task(self):
        task_idx = np.random.randint(0, self.num_tasks)
        return self.tasks[task_idx]

    def adjust_initObjPos(self, orig_init_pos):
        #This is to account for meshes for the geom and object are not aligned
        #If this is not done, the object could be initialized in an extreme position
        diff = self.get_body_com('obj')[:2] - self.data.get_geom_xpos('objGeom')[:2]
        adjustedPos = orig_init_pos[:2] + diff

        #The convention we follow is that body_com[2] is always 0, and geom_pos[2] is the object height
        return [adjustedPos[0], adjustedPos[1],self.data.get_geom_xpos('objGeom')[-1]]


    def reset_model(self):
        self._reset_hand()
        task = self.sample_task()
        self._state_goal = self.sim.model.site_pos[self.model.site_name2id('goal')]
        # self.obj_init_pos = self.adjust_initObjPos(task['obj_init_pos'])
        # self.obj_init_angle = task['obj_init_angle']
        # self.objHeight = self.data.get_geom_xpos('objGeom')[2]
        # self.heightTarget = self.objHeight + self.liftThresh
        goal_pos = np.array([0., 0.4, -0.12])
        # self.obj_init_pos = np.concatenate((self.obj_ [self.obj_init_pos[-1]]))
        self._state_goal = self.sim.model.site_pos[self.model.site_name2id('goal')]
        self._set_goal_marker(self._state_goal)
        # self._set_obj_xyz(self.obj_init_pos)
        #self._set_obj_xyz_quat(self.obj_init_pos, self.obj_init_angle)
        self.curr_path_length = 0
        # self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) - np.array(self._state_goal)) + self.heightTarget
        #Can try changing this
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

    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def compute_rewards(self, actions, obsBatch):
        #Required by HER-TD3
        assert isinstance(obsBatch, dict) == True
        obsList = obsBatch['state_observation']
        rewards = [self.compute_reward(action, obs)[0] for  action, obs in zip(actions, obsList)]
        return np.array(rewards)

    def compute_reward(self, actions, obs, mode = 'general'):
        if isinstance(obs, dict):
            obs = obs['state_observation']


        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        # heightTarget = self.heightTarget
        placingGoal = self._state_goal

        reachDist = np.linalg.norm(placingGoal - fingerCOM)
      

        def reachReward():
            reachDistxy = np.linalg.norm(placingGoal[:-1] - fingerCOM[:-1])
            zRew = np.linalg.norm(fingerCOM[-1] - placingGoal[-1])

            reachRew = -reachDist# + min(actions[-1], -1)/50
            if reachDistxy < 0.05: #0.02
                reachRew = -reachDist
            else:
                reachRew =  -reachDistxy - 2*zRew
            #incentive to close fingers when reachDist is small
            if reachDist < 0.05:
                reachRew = -reachDist + max(actions[-1],0)/50
            return reachRew , reachDist

        reachRew, reachDist = reachReward()
        reward = reachRew
        return reward

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        return statistics

    def log_diagnostics(self, paths = None, logger = None):
        pass


if __name__ == '__main__':
    import time
    env = SawyerHandInsert6DOFEnv()
    for _ in range(1000):
        env.reset()
        # for _ in range(10):
        #     env.data.set_mocap_pos('mocap', np.array([0, 0.8, 0.05]))
        #     env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        #     env.do_simulation([-1,1], env.frame_skip)
        #     #self.do_simulation(None, self.frame_skip)
        # for _ in range(10):
        #     env.data.set_mocap_pos('mocap', np.array([0, 0.8, 0.25]))
        #     # env.data.set_mocap_pos('mocap', np.array([0, 0.6, 0.25]))
        #     env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        #     env.do_simulation([-1,1], env.frame_skip)
        #     #self.do_simulation(None, self.frame_skip)
        for _ in range(100):

            env.render()
            env.step(env.action_space.sample())
            # if _ < 10:
            #     env.step(np.array([0, 0, -1, 0, 0]))
            # elif _ < 50:
            #     env.step(np.array([0, 0, 0, 0, 1]))
            # if _ < 10:
            #     env.step(np.array([0, 0, -1, 0, 0]))
            # else:
            #     env.step(np.array([0, 1, 0, 0, 1]))
                # env.step(np.array([0, 1, 0, 0, 0]))
            # env.step(np.array([np.random.uniform(low=-1., high=1.), np.random.uniform(low=-1., high=1.), 0.]))
            time.sleep(0.05)