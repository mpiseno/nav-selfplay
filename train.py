import numpy as np
import matplotlib.pyplot as plt

import torch
import habitat
import pdb
from habitat_baselines.rl.ppo import PPO
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.utils import batch_obs

from envs import HabitatEnv
#from agents import AliceBobAgent
from models import NavResNetPolicy


class Trainer:
    def __init__(self, config):
        self.config = config
        self.ppo_cfg = config.RL.PPO
        self.static_sensor_uuid = "static"
        self.env = HabitatEnv(self.config)
        self.init_policy_and_agents(self.ppo_cfg)
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.rollouts_alice = RolloutStorage(
            self.ppo_cfg.num_steps,
            num_envs=1,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            recurrent_hidden_state_size=self.ppo_cfg.hidden_size,
        )

        self.rollouts_bob = RolloutStorage(
            self.ppo_cfg.num_steps,
            num_envs=1,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            recurrent_hidden_state_size=self.ppo_cfg.hidden_size,
        )

        self.alice_rewards = []
        self.bob_rewards = []

    def train(self):
        obs = self.env.reset()
        obs[self.static_sensor_uuid] = obs['gps']
        batch = batch_obs([obs], device=self.device)

        # Insert the first observation into Alice's rollout storage
        for sensor in self.rollouts_alice.observations:
            self.rollouts_alice.observations[sensor][0].copy_(batch[sensor])

        # Insert the first observation into the Bob's rollout storage
        for sensor in self.rollouts_bob.observations:
            self.rollouts_bob.observations[sensor][0].copy_(batch[sensor])

        batch = None
        obs = None

        # Main training loop
        #for update in range(self.config.NUM_UPDATES):
        for update in range(1):
            t_alice = 0
            t_max = self.ppo_cfg.num_steps
            obs = self.env.reset()
            start_state = torch.from_numpy(obs['gps'])
            #print(f"interation: {update}, Alice: {obs['gps']}")

            # Alice's turn
            while True:
                goal_state, alice_done = self._alice_training_step(self.rollouts_alice, t_alice, t_max, start_state)

                if alice_done:
                    print("Alice done!")
                    break

                t_alice += 1

            #goal_state = torch.from_numpy(goal_state).float()
            print(f"goal state: {goal_state}")
            # Bob's turn
            obs = self.env.reset()
            t_bob = 0
            while t_alice + t_bob < t_max + 1:
                bob_done = self._bob_training_step(self.rollouts_bob, goal_state, t_bob, t_alice,
                    t_max,
                )

                if bob_done:
                    print("Bob done!")
                    break

                t_bob += 1

            # Calculate episode reward
            self._compute_rewards(t_alice, t_bob)

            # Policy update
            value_loss, action_loss, dist_entropy = self._update_agent(self.ppo_cfg, self.rollouts_alice, self.actor_critic_alice, self.agent_alice)
            value_loss, action_loss, dist_entropy = self._update_agent(self.ppo_cfg, self.rollouts_bob, self.actor_critic_bob, self.agent_bob)

        # a, = plt.plot(np.arange(100) , self.alice_rewards, label="Alice")
        # b, = plt.plot(np.arange(100) , self.bob_rewards, label="Bob")
        # plt.legend(handles=[a, b])
        # plt.show()

    def _alice_training_step(self, rollouts, t_alice, t_max, start_state):
        # Collect the current observation from alice's rollout memory
        with torch.no_grad():
            step_obs = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }

            values, actions, actions_log_probs, hidden_states = self.actor_critic_alice.act(
                step_obs,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            )

        action = actions[0][0].item() # until multiple parallel envs setup, have to index in
        obs, rewards, dones, infos = self.env.step(action, alice=True, train=True)
        obs['static'] = start_state

        # Update alice's rollout memory
        batch = batch_obs([obs], device=self.device)
        rewards = torch.tensor([rewards], dtype=torch.float)
        rewards = rewards.unsqueeze(1)
        dones = [dones]

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float
        )

        rollouts.insert(
            batch,
            hidden_states,
            actions,
            actions_log_probs,
            values,
            rewards,
            masks,
        )

        # Either alice output STOP action or alice timed out
        goal_state = None
        alice_done = False
        if dones[0] or t_alice >= t_max: # 0 is the STOP action
            goal_state = torch.from_numpy(obs['gps'].squeeze()).float()
            alice_done = True

        print(f"Alice: time: {t_alice}, before state: {step_obs['gps']}, action taken: {action}, new state: {obs['gps']}, reward: {rewards[0]}, done: {dones[0]}")

        return goal_state, alice_done

    def _bob_training_step(self, rollouts, goal_state, t_bob, t_alice, t_max):
        with torch.no_grad():
            step_obs = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }

            # bob_cur_state = step_obs['gps']
            # #print(f"t_bob: {t_bob}, goal_state: {goal_state}, bob_state: {bob_cur_state}")
            # bob_done = False
            # if (self._reached_goal(bob_cur_state, goal_state) and t_bob > 0) or t_alice + t_bob > t_max:
            #     bob_done = True
            #     return bob_done

            # TODO: Might change policy to take in start and current state
            values, actions, actions_log_probs, hidden_states = self.actor_critic_bob.act(
                step_obs,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step]
            )

        action = actions[0][0].item()
        obs, rewards, dones, info = self.env.step(action, static_state=goal_state, alice=False, train=True)
        obs['static'] = goal_state

        # Update Bob's rollout memory
        batch = batch_obs([obs], device=self.device)
        rewards = torch.tensor([rewards], dtype=torch.float)
        rewards = rewards.unsqueeze(1)
        dones = [dones]

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float
        )

        rollouts.insert(
            batch,
            hidden_states,
            actions,
            actions_log_probs,
            values,
            rewards,
            masks,
        )

        bob_done = False
        if dones[0] or t_alice + t_bob >= t_max:
            bob_done = True

        print(f"Bob: time: {t_bob}, before state: {step_obs['gps']}, action taken: {action}, new state: {obs['gps']}, reward: {rewards[0]}, done: {dones[0]}")

        return bob_done

    def _compute_rewards(self, t_a, t_b):
        # Reward formula
        r_alice = self.ppo_cfg.gamma_alice_bob * max(0, t_b - t_a)
        r_bob = -self.ppo_cfg.gamma_alice_bob * t_b

        print(f"alice reward: {r_alice}")
        print(f"bob reward: {r_bob}")

        # Insert the rewards into the rollout memory
        alice_step = self.rollouts_alice.step
        self.rollouts_alice.rewards[alice_step - 1, 0, 0] = r_alice
        bob_step = self.rollouts_bob.step
        self.rollouts_bob.rewards[bob_step - 1, 0, 0] = r_bob

        self.alice_rewards.append(r_alice)
        self.bob_rewards.append(r_bob)

    def _update_agent(self, ppo_cfg, rollouts, actor_critic, agent):
        #t_update_model = time.time()
        with torch.no_grad():
            last_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }

            next_value = actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step]
            ).detach()

        rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        return (
            value_loss,
            action_loss,
            dist_entropy,
        )


    def test(self):
        pass

    def save_checkpoint(self):
        pass

    def load_checkpoint(self):
        pass

    def init_policy_and_agents(self, ppo_cfg):
        # self.actor_critic_alice = ObjectNavBaselinePolicy(
        #     observation_space=self.env.observation_space,
        #     action_space=self.env.action_space,
        #     hidden_size=self.ppo_cfg.hidden_size,
        # )
        # self.actor_critic_bob = ObjectNavBaselinePolicy(
        #     observation_space=self.env.observation_space,
        #     action_space=self.env.action_space,
        #     hidden_size=self.ppo_cfg.hidden_size,
        # )

        self.actor_critic_alice = NavResNetPolicy(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            hidden_size=ppo_cfg.hidden_size,
            rnn_type=self.config.RL.PPO.rnn_type,
            num_recurrent_layers=self.config.RL.PPO.num_recurrent_layers,
            backbone=self.config.RL.PPO.backbone,
            normalize_visual_inputs=True
        )
        self.actor_critic_bob = NavResNetPolicy(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            hidden_size=ppo_cfg.hidden_size,
            rnn_type=self.config.RL.PPO.rnn_type,
            num_recurrent_layers=self.config.RL.PPO.num_recurrent_layers,
            backbone=self.config.RL.PPO.backbone,
            normalize_visual_inputs=True
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