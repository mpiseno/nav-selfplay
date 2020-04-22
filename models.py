import torch

from habitat_baselines.rl.ppo import Policy, Net
from habitat_baselines.common.utils import CategoricalNet, Flatten
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.models.simple_cnn import SimpleCNN


class ObjectNavBaselinePolicy(Policy):
    def __init__(self, observation_space, action_space, hidden_size):
        # self.observation_space = observation_space
        # self.action_space = action_space
        # self.hidden_size = hidden_size
        super().__init__(
            ObjectNavBaselineNet(
                observation_space,
                hidden_size
            ),
            action_space.n
        )


class ObjectNavBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, observation_space, hidden_size):
        super().__init__()
        #self.goal_sensor_uuid = goal_sensor_uuid
        # self._n_input_goal = observation_space.spaces[
        #     self.goal_sensor_uuid
        # ].shape[0]
        self._hidden_size = hidden_size

        self.visual_encoder = SimpleCNN(observation_space, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self._hidden_size),
            self._hidden_size,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_encoding(self, observations):
        return observations[self.goal_sensor_uuid]

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        #target_encoding = self.get_target_encoding(observations)
        #x = [target_encoding]
        x = []

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            x = [perception_embed] + x

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states

