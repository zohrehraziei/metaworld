import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
from collections import deque
import gym
from gym.spaces import Box
from multiworld.envs.mujoco.sawyer_xyz.procedural.random_hinges import HingeEnv, DoorSide
import numpy as np
import pdb

try:
	import mujoco_py
except ImportError as e:
	raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 500

# ENV_LIST = [SawyerReachPushPickPlace6DOFEnv, SawyerReachPushPickPlace6DOFEnv, SawyerReachPushPickPlace6DOFEnv, SawyerShelfPlace6DOFEnv, SawyerDrawerOpen6DOFEnv,
# 			SawyerDrawerClose6DOFEnv, SawyerButtonPress6DOFEnv, SawyerButtonPressTopdown6DOFEnv, SawyerPegInsertionSide6DOFEnv]

NUM_ENVS = 10
ENV_LIST = []

np.random.seed(42)

for i in range(NUM_ENVS):
	idx = np.random.choice([0, 1])
	sign = np.random.choice([-1, 1])
	hinge_sign = np.random.choice([-1, 1])
	task_idx = np.zeros(NUM_ENVS)
	task_idx[i] = 1
	env = HingeEnv(DoorSide(idx, sign, hinge_sign), rotMode='rotz', num_tasks=NUM_ENVS, task_idx=task_idx)
	ENV_LIST.append(env)
# print(ENV_LIST)

class  ProceduralMultitaskEnv(gym.Env):
	"""
	An multitask mujoco environment that contains a list of mujoco environments.
	"""
	def __init__(self,
				if_render=False,
				adaptive_sampling=False):
		self.mujoco_envs = []
		self.mujoco_envs = ENV_LIST
		self.adaptive_sampling = adaptive_sampling
		self.sample_probs = None
		if adaptive_sampling:
			self.scores = {i:deque(maxlen=10) for i in range(len(ENV_LIST))}
			self.current_score = 0.
			self.sample_probs = np.array([1./(len(ENV_LIST)) for _ in range(len(ENV_LIST))])
			self.target_scores = []
			self.sample_tau = 0.05
		for i, env in enumerate(ENV_LIST):
			# if i < 3:
			# 	self.mujoco_envs.append(env(multitask=True, multitask_num=len(ENV_LIST), random_init=False, if_render=if_render, fix_task=True, task_idx=i))
			# else:
			# 	self.mujoco_envs.append(env(multitask=True, multitask_num=len(ENV_LIST), random_init=False, if_render=if_render))
			# set the one-hot task representation
			self.mujoco_envs[i]._state_goal_idx = np.zeros((len(ENV_LIST)))
			self.mujoco_envs[i]._state_goal_idx[i] = 1.
			if adaptive_sampling:
				self.target_scores.append(self.mujoco_envs[i].target_reward)
		# TODO: make sure all observation spaces across tasks are the same / use self-attention
		self.num_resets = 0
		self.task_idx = self.num_resets % len(ENV_LIST)
		self.action_space = self.mujoco_envs[self.task_idx].action_space
		self.observation_space = self.mujoco_envs[self.task_idx].observation_space
		self.goal_space = Box(np.zeros(len(ENV_LIST)), np.ones(len(ENV_LIST)))
		self.reset()
		self.num_resets -= 1

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action):
		ob, reward, done, info = self.mujoco_envs[self.task_idx].step(action)
		# TODO: need to figure out how to calculate target scores in this case!
		if self.adaptive_sampling:
			self.current_score += reward
			if done:
				self.scores[self.task_idx].append(reward)
				# self.scores[self.task_idx].append(self.current_score)
		return ob, reward, done, info

	def reset(self):
		# self.task_idx = max((self.num_resets % (len(ENV_LIST)+2)) - 2, 0)
		if self.num_resets < len(ENV_LIST)*2 or not self.adaptive_sampling:
			self.task_idx = self.num_resets % len(ENV_LIST)
		else:
			if self.num_resets % (8*len(ENV_LIST)) == 0:
				avg_scores = [np.mean(self.scores[i]) for i in range(len(ENV_LIST))]
				self.sample_probs = np.exp((np.array(self.target_scores) - np.array(avg_scores))/np.array(self.target_scores)/self.sample_tau)
				self.sample_probs = self.sample_probs / self.sample_probs.sum()
				print('Sampling prob is', self.sample_probs)
			self.task_idx = np.random.choice(range(len(ENV_LIST)), p=self.sample_probs)
		self.num_resets += 1
		self.current_score = 0.
		return self.mujoco_envs[self.task_idx].reset()

	@property
	def dt(self):
		return self.mujoco_envs[self.task_idx].model.opt.timestep * self.mujoco_envs[self.task_idx].frame_skip

	def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE, depth=False):
		return self.mujoco_envs[self.task_idx].render(mode=mode, width=width, height=height, depth=depth)

	def close(self):
		self.mujoco_envs[self.task_idx].close()
