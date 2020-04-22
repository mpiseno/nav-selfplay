import torch
import habitat
from habitat_baselines.rl.ppo import PPO
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.utils import batch_obs

from envs import HabitatEnv
#from agents import AliceBobAgent
from models import ObjectNavBaselinePolicy


class Trainer:
    def __init__(self, config):
        self.config = config
        self.ppo_cfg = config.RL.PPO
        self.env = HabitatEnv(self.config)
        self.init_policy_and_agents(self.ppo_cfg)
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.rollouts = RolloutStorage(
            self.ppo_cfg.num_steps,
            num_envs=1,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            recurrent_hidden_state_size=self.ppo_cfg.hidden_size,
        )

    def train(self):
        observations = [self.env.reset()]
        batch = batch_obs(observations, device=self.device)

        # Insert the first observation into the rollout storage
        for sensor in self.rollouts.observations:
            self.rollouts.observations[sensor][0].copy_(batch[sensor])

        batch = None
        observations = None

        # Main training loop
        #for update in range(self.config.NUM_UPDATES):
        for update in range(1):
            t_alice = 0
            t_max = self.ppo_cfg.num_steps
            obs = self.env.reset()
            start_state = obs['gps']

            # Alice's turn
            while True:
                t_alice += 1

                step_obs = {
                    k: v[self.rollouts.step] for k, v in self.rollouts.observations.items()
                }

                # TODO: Might change policy to take in start and current state
                values, actions, actions_log_probs, hidden_states = self.actor_critic_alice.act(
                    step_obs,
                    self.rollouts.recurrent_hidden_states[self.rollouts.step],
                    self.rollouts.prev_actions[self.rollouts.step],
                    self.rollouts.masks[self.rollouts.step],
                )

                action = actions[0][0] # until multiple parallel envs setup, have to index in
                print(action)
                # We reached the goal
                if action == 0 or t_alice > t_max:
                    goal_state = obs['gps']
                    break

                obs = self.env.step(action)

            obs = self.env.reset()
            t_bob = 0
            # Bob's turn
            while True:
                bob_cur_state = obs['gps']
                if bob_cur_state == s_goal or t_alice + t_bob > t_max:
                    break

                t_bob += 1
                step_obs = {
                    k: v[self.rollouts.step] for k, v in self.rollouts.observations.items()
                }

                # TODO: Might change policy to take in start and current state
                values, actions, actions_log_probs, hidden_states = self.actor_critic_bob.act(
                    step_obs,
                    self.rollouts.recurrent_hidden_states[self.rollouts.step],
                    self.rollouts.prev_actions[self.rollouts.step],
                    self.rollouts.masks[self.rollouts.step],
                )

                action = actions[0][0]
                obs = self.env.step(action)

            # Calculate episode reward
            R_alice = ppo_cfg.gamma_alice_bob * max(0, t_bob - t_alice)
            R_bob = -ppo_cfg.gamma_alice_bob * t_bob
            

    def test(self):
        pass

    def save_checkpoint(self):
        pass

    def load_checkpoint(self):
        pass

    def init_policy_and_agents(self, ppo_cfg):
        self.actor_critic_alice = ObjectNavBaselinePolicy(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            hidden_size=self.ppo_cfg.hidden_size,
        )
        self.actor_critic_bob = ObjectNavBaselinePolicy(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            hidden_size=self.ppo_cfg.hidden_size,
        )

        self.agent_alice = PPO(
            actor_critic=self.actor_critic_alice,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            #use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )

        self.agent_bob = PPO(
            actor_critic=self.actor_critic_bob,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            #use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )