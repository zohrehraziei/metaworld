"""Proposal for a simple, understandable MetaWorld API."""
from collections import namedtuple
from typing import OrderedDict, List, Type

import gym

EnvID = str

class Task(namedtuple):
    env_id: EnvID
    data: bytes  # Contains env parameters like random_init and *a* goal


class MetaWorldEnv(gym.Env):
    """Environment that requires a task before use.

    Takes no arguments to its constructor, and raises an exception if used
    before `set_task` is called.
    """

    def set_task(self, task: Task) -> None:
        """Set the task.

        Raises:
            ValueError: If task.env_id is different from the current task.

        """


class Benchmark:

    @classmethod
    def get_train_classes(cls) -> OrderedDict[EnvID, Type]:
        """Get all of the environment classes used for training."""

    @classmethod
    def get_test_classes(cls) -> OrderedDict[EnvID, Type]:
        """Get all of the environment classes used for testing."""

    @classmethod
    def get_train_tasks(cls) -> List[Task]:
        """Get all of the training tasks for this benchmark."""

    @classmethod
    def get_test_tasks(cls) -> List[Task]:
        """Get all of the test tasks for this benchmark."""
