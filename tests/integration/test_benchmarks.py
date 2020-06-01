import numpy as np
import pytest

from metaworld.benchmarks import ML1, MT10, ML10, ML45, MT50
from metaworld.envs.mujoco.env_dict import ALL_ENVIRONMENTS
from tests.helpers import step_env


# @pytest.mark.parametrize('name', ML1.available_tasks())
# def test_all_ml1(name):
#     train_env = ML1.get_train_tasks(name)
#     tasks = train_env.sample_tasks(11)
#     for t in tasks:
#         train_env.set_task(t)
#         step_env(train_env, max_path_length=3)

#     train_env.close()
#     del train_env

#     test_env = ML1.get_test_tasks(name)
#     tasks = test_env.sample_tasks(11)
#     for t in tasks:
#         test_env.set_task(t)
#         step_env(test_env, max_path_length=3)

#     test_env.close()
#     del test_env


# def test_all_ml10():
#     ml10_train_env = ML10.get_train_tasks()
#     train_tasks = ml10_train_env.sample_tasks(11)
#     for t in train_tasks:
#         ml10_train_env.set_task(t)
#         step_env(ml10_train_env, max_path_length=3)

#     ml10_train_env.close()
#     del ml10_train_env

#     ml10_test_env = ML10.get_test_tasks()
#     test_tasks = ml10_test_env.sample_tasks(11)
#     for t in test_tasks:
#         ml10_test_env.set_task(t)
#         step_env(ml10_test_env, max_path_length=3)

#     ml10_test_env.close()
#     del ml10_test_env


# def test_all_mt10():
#     mt10_env = MT10()
#     tasks = mt10_env.sample_tasks(11)
#     for t in tasks:
#         mt10_env.set_task(t)
#         step_env(mt10_env, max_path_length=3)

#     mt10_env.close()
#     del mt10_env


# @pytest.mark.large
# def test_all_ml45():
#     ml45_train_env = ML45.get_train_tasks()
#     train_tasks = ml45_train_env.sample_tasks(46)
#     for t in train_tasks:
#         ml45_train_env.set_task(t)
#         step_env(ml45_train_env, max_path_length=3)

#     ml45_train_env.close()
#     del ml45_train_env

#     ml45_test_env = ML45.get_test_tasks()
#     test_tasks = ml45_test_env.sample_tasks(6)
#     for t in test_tasks:
#         ml45_test_env.set_task(t)
#         step_env(ml45_test_env, max_path_length=3)

#     ml45_test_env.close()
#     del ml45_test_env


# @pytest.mark.large
# def test_all_mt50():
#     mt50_env = MT50()
#     tasks = mt50_env.sample_tasks(51)
#     for t in tasks:
#         mt50_env.set_task(t)
#         step_env(mt50_env, max_path_length=3)

#     mt50_env.close()
#     del mt50_env

@pytest.mark.parametrize('env_cls', list(ALL_ENVIRONMENTS.keys()))  # ['push-v1', 'reach-v1']
def test_valid_goal(env_cls):
    """Verify that the goal set by env is reachable by the end effector."""
    task_types = ['pick-place', 'reach', 'push']
    task_type = None
    for name in task_types:
        if name in env_cls:
            task_type = name
        if task_type == 'pick-place':
            task_type = 'pick_place'
    if env_cls in ['push-back-v1', 'stick-push-v1']:
        task_type = None
    if task_type:
        env = ALL_ENVIRONMENTS[env_cls](random_init=True, task_type=task_type)
    else:
        env = ALL_ENVIRONMENTS[env_cls](random_init=True)

    for _ in range(500):
        goal = env._state_goal
        env.data.set_mocap_pos('mocap', goal)
        inital_endeff_pos = env.get_endeff_pos()
        env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        assert (env.data.get_mocap_pos('mocap') == goal).all()
        for _ in range(1000):
            env.sim.step()
        assert np.isclose(goal, env.get_endeff_pos(), atol=0.05).all()
        env.reset()
    del env