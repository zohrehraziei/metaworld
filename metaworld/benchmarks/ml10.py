
from metaworld.benchmarks.base import Benchmark
from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv
from metaworld.envs.mujoco.env_dict import MEDIUM_MODE_ARGS_KWARGS, MEDIUM_MODE_CLS_DICT


class ML10(MultiClassMultiTaskEnv, Benchmark):

    def __init__(self, env_type='train', sample_all=False, task_names=None):
        assert env_type == 'train' or env_type == 'test'

        if task_names is not None:
            for task_name in task_names:
                if task_name not in MEDIUM_MODE_CLS_DICT[env_type]:
                    raise ValueError("{} does not exist in ML10 {} tasks".format(
                        task_name, env_type))
            cls_dict = {task_name: MEDIUM_MODE_CLS_DICT[env_type][task_name] for task_name in task_names}
            args_kwargs = {task_name: MEDIUM_MODE_ARGS_KWARGS[env_type][task_name] for task_name in task_names}
        else:
            cls_dict = MEDIUM_MODE_CLS_DICT[env_type]
            args_kwargs = MEDIUM_MODE_ARGS_KWARGS[env_type]

        super().__init__(
            task_env_cls_dict=cls_dict,
            task_args_kwargs=args_kwargs,
            sample_goals=True,
            obs_type='plain',
            sample_all=sample_all)

    @classmethod
    def from_task(cls, task_name):
        return cls.from_tasks([task_name])
            
    @classmethod
    def from_tasks(cls, task_names):
        assert task_names
        
        if task_names[0] in MEDIUM_MODE_CLS_DICT['train']:
            return cls(env_type='train', sample_all=True, task_names=task_names)
        elif task_names[0] in MEDIUM_MODE_CLS_DICT['test']:
            return cls(env_type='test', sample_all=True, task_names=task_names)
        else:
            raise ValueError('{} does not exist in ML10'.format(task_name[0]))
            
