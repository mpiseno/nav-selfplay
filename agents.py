import random

import habitat
import torch
import torch.nn as nn

# from habitat_baselines.common.utils import (
#     CategoricalNet,
#     Flatten,
#     ResizeCenterCropper,
# )
# from habitat_baselines.rl.ddppo.policy import resnet
# from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
# from habitat_baselines.rl.ppo import Net, Policy


class AliceBobAgent(habitat.core.Agent):
    def __init__(self, config):
        self.alice = TrivialPolicy(**config)
        self.bob = TrivialPolicy(**config)
        self.is_alice = True

    def act(self, observations, rnn_hidden_states, prev_actions, masks):
        # Alice's turn
        if self.is_alice:
            pass

        # Bob's turn
        else:
            pass

        return None



# class NavPolicy(Policy):
#     def __init__(
#         self,
#         observation_space,
#         action_space,
#         #goal_sensor_uuid="pointgoal_with_gps_compass",
#         hidden_size=512,
#         num_recurrent_layers=2,
#         rnn_type="LSTM",
#         resnet_baseplanes=32,
#         backbone="resnet50",
#         normalize_visual_inputs=False,
#         obs_transform=ResizeCenterCropper(size=(256, 256)),
#     ):
#         super().__init__(
#             NavResNetNet(
#                 observation_space=observation_space,
#                 action_space=action_space,
#                 hidden_size=hidden_size,
#                 num_recurrent_layers=num_recurrent_layers,
#                 rnn_type=rnn_type,
#                 backbone=backbone,
#                 resnet_baseplanes=resnet_baseplanes,
#                 normalize_visual_inputs=normalize_visual_inputs,
#                 obs_transform=obs_transform,
#             ),
#             action_space.n,
#         )


# class NavResNetNet(Net):
#     def __init__(
#         self,
#         observation_space,
#         action_space,
#         hidden_size,
#         num_recurrent_layers,
#         rnn_type,
#         backbone,
#         resnet_baseplanes,
#         normalize_visual_inputs,
#         obs_transform=ResizeCenterCropper(size=(256, 256)),
#     ):
#         # Takes the previous action as input
#         self.prev_action_embedding = nn.Embedding(action_space.n, 32)
#         self._n_prev_action = 32

#         # Visual encoder takes the rgb+depth observation
#         self.visual_encoder = ResNetEncoder(
#             observation_space,
#             baseplanes=resnet_baseplanes,
#             ngroups=resnet_baseplanes // 2,
#             make_backbone=getattr(resnet, backbone),
#             normalize_visual_inputs=normalize_visual_inputs,
#             obs_transform=obs_transform,
#         )

#         if not self.visual_encoder.is_blind:
#             self.visual_fc = nn.Sequential(
#                 Flatten(),
#                 nn.Linear(
#                     np.prod(self.visual_encoder.output_shape), hidden_size
#                 ),
#                 nn.ReLU(True),
#             )

#         self.state_encoder = RNNStateEncoder(
#             (0 if self.is_blind else self._hidden_size) + self._n_prev_action,
#             self._hidden_size,
#             rnn_type=rnn_type,
#             num_layers=num_recurrent_layers,
#         )

#         self.train()   

#     @property
#     def output_size(self):
#         return self._hidden_size

#     @property
#     def is_blind(self):
#         return self.visual_encoder.is_blind

#     @property
#     def num_recurrent_layers(self):
#         return self.state_encoder.num_recurrent_layers

#     # def get_tgt_encoding(self, observations):
#     #     goal_observations = observations[self.goal_sensor_uuid]
#     #     goal_observations = torch.stack(
#     #         [
#     #             goal_observations[:, 0],
#     #             torch.cos(-goal_observations[:, 1]),
#     #             torch.sin(-goal_observations[:, 1]),
#     #         ],
#     #         -1,
#     #     )

#     #     return self.tgt_embeding(goal_observations)

#     def forward(self, observations, rnn_hidden_states, prev_actions, masks):
#         x = []
#         if not self.is_blind:
#             if "visual_features" in observations:
#                 visual_feats = observations["visual_features"]
#             else:
#                 visual_feats = self.visual_encoder(observations)

#             visual_feats = self.visual_fc(visual_feats)
#             x.append(visual_feats)

#         #tgt_encoding = self.get_tgt_encoding(observations)
#         prev_actions = self.prev_action_embedding(
#             ((prev_actions.float() + 1) * masks).long().squeeze(-1)
#         )

#         x += [tgt_encoding, prev_actions]

#         x = torch.cat(x, dim=1)
#         x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

#         return x, rnn_hidden_states


# '''
# Code taken from https://github.com/facebookresearch/habitat-api/blob/c2473c2afbe0d5db4ab4b463d58455bb4ac2e66e/habitat_baselines/rl/ddppo/policy/resnet_policy.py
# '''
# class ResNetEncoder(nn.Module):
#     def __init__(
#         self,
#         observation_space,
#         baseplanes=32,
#         ngroups=32,
#         spatial_size=128,
#         make_backbone=None,
#         normalize_visual_inputs=False,
#         obs_transform=ResizeCenterCropper(size=(256, 256)),
#     ):
#         super().__init__()

#         self.obs_transform = obs_transform
#         if self.obs_transform is not None:
#             observation_space = self.obs_transform.transform_observation_space(
#                 observation_space
#             )

#         if "rgb" in observation_space.spaces:
#             self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
#             spatial_size = observation_space.spaces["rgb"].shape[0] // 2
#         else:
#             self._n_input_rgb = 0

#         if "depth" in observation_space.spaces:
#             self._n_input_depth = observation_space.spaces["depth"].shape[2]
#             spatial_size = observation_space.spaces["depth"].shape[0] // 2
#         else:
#             self._n_input_depth = 0

#         if normalize_visual_inputs:
#             self.running_mean_and_var = RunningMeanAndVar(
#                 self._n_input_depth + self._n_input_rgb
#             )
#         else:
#             self.running_mean_and_var = nn.Sequential()

#         if not self.is_blind:
#             input_channels = self._n_input_depth + self._n_input_rgb
#             self.backbone = make_backbone(input_channels, baseplanes, ngroups)

#             final_spatial = int(
#                 spatial_size * self.backbone.final_spatial_compress
#             )
#             after_compression_flat_size = 2048
#             num_compression_channels = int(
#                 round(after_compression_flat_size / (final_spatial ** 2))
#             )
#             self.compression = nn.Sequential(
#                 nn.Conv2d(
#                     self.backbone.final_channels,
#                     num_compression_channels,
#                     kernel_size=3,
#                     padding=1,
#                     bias=False,
#                 ),
#                 nn.GroupNorm(1, num_compression_channels),
#                 nn.ReLU(True),
#             )

#             self.output_shape = (
#                 num_compression_channels,
#                 final_spatial,
#                 final_spatial,
#             )

#     @property
#     def is_blind(self):
#         return self._n_input_rgb + self._n_input_depth == 0

#     def layer_init(self):
#         for layer in self.modules():
#             if isinstance(layer, (nn.Conv2d, nn.Linear)):
#                 nn.init.kaiming_normal_(
#                     layer.weight, nn.init.calculate_gain("relu")
#                 )
#                 if layer.bias is not None:
#                     nn.init.constant_(layer.bias, val=0)

#     def forward(self, observations):
#         if self.is_blind:
#             return None

#         cnn_input = []
#         if self._n_input_rgb > 0:
#             rgb_observations = observations["rgb"]
#             # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
#             rgb_observations = rgb_observations.permute(0, 3, 1, 2)
#             rgb_observations = rgb_observations / 255.0  # normalize RGB
#             cnn_input.append(rgb_observations)

#         if self._n_input_depth > 0:
#             depth_observations = observations["depth"]

#             # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
#             depth_observations = depth_observations.permute(0, 3, 1, 2)

#             cnn_input.append(depth_observations)

#         if self.obs_transform:
#             cnn_input = [self.obs_transform(inp) for inp in cnn_input]

#         x = torch.cat(cnn_input, dim=1)
#         x = F.avg_pool2d(x, 2)

#         x = self.running_mean_and_var(x)
#         x = self.backbone(x)
#         x = self.compression(x)
#         return x
