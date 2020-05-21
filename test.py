from metaworld.envs.mujoco.env_dict import ALL_ENVIRONMENTS
import time

curr = time.time()
env = ALL_ENVIRONMENTS['push-v1'](random_init=False, task_type='push')
curr_time_step = 1
while 1:
    env.step(env.action_space.sample())
    env.render()
    if curr_time_step == env.max_path_length:
        curr_time_step = 0
        env.reset()
    curr_time_step += 1

