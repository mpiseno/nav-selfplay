import habitat
import gym
from gym.spaces.dict_space import Dict as SpaceDict
import numpy as np


class HabitatEnv(habitat.RLEnv):
    def __init__(self, config):
        super().__init__(config)
        self.observation_space = SpaceDict(
            {
                **self._env._sim.sensor_suite.observation_spaces.spaces,
                **self._env._task.sensor_suite.observation_spaces.spaces,
                "static": gym.spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(2,)
                )
            }
        )

    def observe(self):
        observations = self._env.step("STOP")
        self._env._episode_over = False
        return observations

    def get_reward_range(self):
        r"""Get min, max range of reward.

        :return: :py:`[min, max]` range of reward.
        """
        return [0, 1]

    def get_reward(self, observations):
        r"""Returns reward after action has been performed.

        :param observations: observations from simulator and task.
        :return: reward after performing the last action.

        This method is called inside the `step()` method.
        """
        return 0

    def get_done(self, observations):
        r"""Returns boolean indicating whether episode is done after performing
        the last action.

        :param observations: observations from simulator and task.
        :return: done boolean after performing the last action.

        This method is called inside the step method.
        """
        return False

    def get_info(self, observations):
        r"""..

        :param observations: observations from simulator and task.
        :return: info after performing the last action.
        """
        return dict()