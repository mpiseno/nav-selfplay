import numpy as np
import torch
import gym
import habitat

from gym.spaces.dict_space import Dict as SpaceDict


class HabitatEnv(habitat.RLEnv):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
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

    def step(self, *args, **kwargs):
        r"""Perform an action in the environment.

        :return: :py:`(observations, reward, done, info)`
        """
        train = kwargs["train"]

        # If we're training, we don't want to interate the episode
        if train:
            self._env._episode_over = False

        observations = self._env.step(*args, **kwargs)
        reward = self.get_reward(observations)
        if train:
            done = self.get_done_train(observations, *args, **kwargs)
        else:
            done = self.get_done_test(observations)
        info = self.get_info(observations)

        return observations, reward, done, info

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

    def get_done_train(self, observations, *args, **kwargs):
        r"""Returns boolean indicating whether episode is done after performing
        the last action.

        :param observations: observations from simulator and task.
        :return: done boolean after performing the last action.

        This method is called inside the step method.
        """
        action, = args
        is_alice = kwargs["alice"]
        done = False
        if is_alice:
            done = action == 0
        else:
            if action == 0:
                cur_state = torch.from_numpy(observations["gps"]).float()
                goal_state = kwargs["static_state"]
                done = self._reached_goal(cur_state, goal_state)

        return done

    def get_done_test(self, observations, *args, **kwargs):
        return False

    def get_info(self, observations):
        r"""..

        :param observations: observations from simulator and task.
        :return: info after performing the last action.
        """
        return dict()

    def _reached_goal(self, bob_cur_state, goal_state):
        return torch.dist(bob_cur_state, goal_state, p=2) <= 0.2
