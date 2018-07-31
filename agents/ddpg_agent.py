import csv
from collections import namedtuple

import numpy as np

from agents.actor_critic import Actor, Critic
from agents.noise import OUNoise
from agents.prioritized_replay_buffer import PrioritizedReplayBuffer
from agents.replay_buffer import ReplayBuffer
from agents.schedules import LinearSchedule


class DDPG():
    """
    Reinforcement Learning agent that learns using DDPG.
    Deep DPG as described by Lillicrap et al. (2015)
    """

    def __init__(self, task, prioritized_replay=True):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15 #0.15 #0.1
        self.exploration_sigma = 0.2 #0.2 #0.1
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        self.buffer_size = 100000
        self.batch_size = 64  # 64

        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_alpha = 0.6
        self.prioritized_replay_beta0 = 0.4
        self.prioritized_replay_beta_iters = None
        self.prioritized_replay_eps = 1e-6
        self.max_timesteps = 100000

        # Replay buffer
        if self.prioritized_replay:
            self.memory = PrioritizedReplayBuffer(self.buffer_size, alpha=self.prioritized_replay_alpha)
            if self.prioritized_replay_beta_iters is None:
                self.prioritized_replay_beta_iters = self.max_timesteps
            self.beta_schedule = LinearSchedule(self.prioritized_replay_beta_iters,
                                           initial_p=self.prioritized_replay_beta0,
                                           final_p=1.0)
        else:
            self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.01  # for soft update of target parameters
        #self.tau = 0.001 # 0.001 per paper

        self.td_errors_list = []
        self.actor_loss_list = []
        self.critic_loss_list = []

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            if self.prioritized_replay:
                samples = self.memory.sample(self.batch_size, beta=self.beta_schedule.value(len(self.memory)))
                (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = samples
                experiences = []
                for i in range(len(obses_t)):
                    experiences.append(namedtuple("PrioritizedExperience", field_names=["state", "action", "reward", "next_state", "done", "weight", "batch_idx"])(obses_t[i:i+1], actions[i:i+1], rewards[i:i+1], obses_tp1[i:i+1], dones[i:i+1], weights[i:i+1], batch_idxes[i:i+1]))
                self.learn(experiences)
            else:
                experiences = self.memory.sample()
                self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]

        #actions = list(action + self.noise.sample())
        #print("act {}".format(actions))
        #return actions  # add some noise for exploration

        return list(action + self.noise.sample())  # add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        # Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        critic_loss = self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local) using action gradients
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        actor_loss = self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

        if self.prioritized_replay:
            # Update replay buffer priorities
            batch_idxes = np.vstack([e.batch_idx[0] for e in experiences if e is not None])
            new_priorities = np.abs(Q_targets) + self.prioritized_replay_eps
            self.memory.update_priorities(batch_idxes, new_priorities)

        self.td_errors_list.append(Q_targets.T)
        self.actor_loss_list.append(actor_loss[0])
        self.critic_loss_list.append(critic_loss)

        #print("states {} next states {} critic_loss {} actor_loss {}".format(states, actions_next, critic_loss, actor_loss))

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

    def save_weights(self):
        self.actor_local.model.save_weights("DDPG_actor_weights.h5")
        self.critic_local.model.save_weights("DDPG_critic_weights.h5")

    def save_td_errors(self, i_episode):
        with open("DDPG_agent_td_errors_episode_{}.csv".format(i_episode), 'w') as csvfile:
            writer = csv.writer(csvfile)
            for td_errors in self.td_errors_list:
                writer.writerow([td_errors])
        self.td_errors_list.clear()

    def save_losses(self, i_episode):
        with open("DDPG_agent_actor_critic_loss_episode_{}.csv".format(i_episode), 'w') as csvfile:
            writer = csv.writer(csvfile)
            for actor_loss, critic_loss in zip(self.actor_loss_list, self.critic_loss_list):
                writer.writerow([actor_loss, critic_loss])

        self.actor_loss_list.clear()
        self.critic_loss_list.clear()
