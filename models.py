import torch
import torch.nn as nn

from habitat_baselines.rl.ppo import Net
from habitat_baselines.rl.ppo.policy import CriticHead
from habitat_baselines.common.utils import CategoricalNet, Flatten
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.models.simple_cnn import SimpleCNN


class Policy(nn.Module):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )

        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states


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
        # self.goal_sensor_uuid = goal_sensor_uuid
        # self._n_input_goal = observation_space.spaces[
        #     self.goal_sensor_uuid
        # ].shape[0]

        # Change this input size if the state representation changes
        self._rnn_input_size = hidden_size + 2 + 2 # hidden size + current state + static state
        self._hidden_size = hidden_size
        self.visual_encoder = SimpleCNN(observation_space, hidden_size)
        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self._rnn_input_size),
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
        current_state = observations['gps']
        static_state = observations['static']
        x = [static_state, current_state]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            x = [perception_embed] + x

        x = torch.cat(x, dim=1)

        print(x.size())
        print(rnn_hidden_states.size())
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states


