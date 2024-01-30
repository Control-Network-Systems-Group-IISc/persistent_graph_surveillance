'''Module for the PPO agent'''
#from gymnasium import spaces
from collections import namedtuple

import torch
from torch.optim import AdamW
from torch.distributions import Categorical
#from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.utils.clip_grad import clip_grad_value_

from models.actor import TorchSingleAgentActor
from models.actor import ActorUtils

from models.critic import TorchSingleAgentCritic
from models.critic import CriticUtils

import random

class PPOAgent:
    '''Class for the PPO Agent'''
    def __init__(self, model_config, hyperparameters):
        # Make sure the environment is compatible with our code
        #assert type(env.observation_space) == spaces.Box
        #assert type(env.action_space) == spaces.Box
        # set to 1 thread
        torch.set_num_threads(1)
        if torch.cuda.is_available():
            print('\n===============Using CUDA=====================\n')
            dev = 'cuda:0'
        else:
            print('\n===============Using CPU=====================\n')
            dev = 'cpu'
        self.device = torch.device(dev)

        self._init_hyperparameters(hyperparameters)

        # Initialize actor and critic networks
        self.actor = TorchSingleAgentActor(**model_config) # ALG STEP 1
        self.actor_utils = ActorUtils(**model_config)

        self.critic = TorchSingleAgentCritic(**model_config)
        self.critic_utils = CriticUtils(**model_config)

        # Initialize optimizers for actor and critic
        self.actor_optim = AdamW(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = AdamW(self.critic.parameters(), lr=self.critic_lr)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
                'actor_loss': [],
                'critic_loss': [],
            }
        self.actor.to(self.device)
        self.critic.to(self.device)

        self.experience = namedtuple('experience',
                        ['state', 'action', 'reward',
                        'log_prob', 'next_state', 'done'])
        self.batch_rollout:list[list[self.experience]] = []

        # adding exploration noise
        self.noise = OUNoise(
                action_dim=model_config['nx_graph'].number_of_nodes()
                )
        self.t = 0

    def new_ep(self):
        self.batch_rollout.append([])

    def update_rollout(self, state, action, log_prob, reward, next_state, done):
        self.batch_rollout[-1].append(self.experience(
            state, action, reward, log_prob, next_state, done
            ))

    def reset_rollout(self):
        self.batch_rollout = []
        self.logger['actor_loss'] = []
        self.logger['critic_loss'] = []

    def load_saved_model(self, actor_path, critic_path):
        self.actor.load_state_dict(torch.load(actor_path,
                                    map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path,
                                    map_location=self.device))

    def get_experiences(self):
        batch_obs = [[]]
        batch_acts = [[]]
        batch_log_probs = [[]]
        batch_rewards = [[]]

        for rollout in self.batch_rollout:
            for experience in rollout:
                batch_obs[-1].append(experience.state)
                batch_acts[-1].append(experience.action)
                batch_rewards[-1].append(experience.reward)
                batch_log_probs[-1].append(experience.log_prob)
            # batch_obs.append([])
            # batch_acts.append([])
            batch_rewards.append([])
            # batch_log_probs.append([])

        # remove the last empty list
        # batch_acts.pop()
        # batch_log_probs.pop()
        batch_rewards.pop()
        # batch_obs.pop()
        # convert to tensors
        batch_rtgs = self.compute_rtgs(batch_rewards)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float32)
        batch_log_probs = torch.tensor(batch_log_probs,
                                    dtype=torch.float32).flatten()
        # Move to cuda if available
        batch_acts = batch_acts.to(self.device)
        batch_log_probs = batch_log_probs.to(self.device)
        batch_rtgs = batch_rtgs.to(self.device)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs

    def learn(self):
        (batch_obs, batch_acts,
        batch_log_probs, batch_rtgs) = self.get_experiences()

        # Calculate advantage at k-th iteration
        values, _ = self.evaluate(batch_obs, batch_acts)
        advantage_k = batch_rtgs - values.detach()

        # Normalize advantages
        advantage_k = (advantage_k -
                advantage_k.mean()) / (advantage_k.std() + 1e-10)

        # This is the loop where we update our network for some n epochs
        for _ in range(self.n_updates_per_iteration):# ALG STEP 6 & 7
            # Calculate V_phi and pi_theta(a_t | s_t)
            values, curr_log_probs = self.evaluate(batch_obs, batch_acts)
            # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
            # NOTE: we just subtract the logs, which is the same as
            # dividing the values and then canceling the log with e^log.
            # For why we use log probabilities instead of actual probabilities,
            # here's a great explanation:
            # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
            # TL;DR makes gradient ascent easier behind the scenes.
            ratios = torch.exp(curr_log_probs - batch_log_probs)

            # Calculate surrogate losses.
            surr1 = ratios * advantage_k
            surr2 = torch.clamp(ratios,
                    1 - self.clip, 1 + self.clip) * advantage_k

            # Calculate actor and critic losses.
            # NOTE: we take the negative min of the surrogate losses coz we're
            # trying to maximize the performance function, but Adam
            # minimizes the loss. So minimizing the negative performance
            # function maximizes it.
            # add entropy value and maximize it
            # subtract curr_log_probs.mean() from actor loss for max entropy RL
            actor_loss = (-torch.min(surr1, surr2)).mean()
            actor_loss -= 0.2*curr_log_probs.mean()
            critic_loss = torch.nn.SmoothL1Loss()(values, batch_rtgs)

            # Calculate gradients and perform backward propagation
            self.actor_optim.zero_grad()
            actor_loss.backward()
            #clip_grad_norm_(self.actor.parameters(), self.grad_clip_value)
            clip_grad_value_(self.actor.parameters(), self.grad_clip_value)
            self.actor_optim.step()

            self.logger['actor_loss'].append(actor_loss.cpu().detach().numpy())

            # Calculate gradients and perform backward propagation for critic
            self.critic_optim.zero_grad()
            critic_loss.backward()
            #clip_grad_norm_(self.critic.parameters(), self.grad_clip_value)
            clip_grad_value_(self.critic.parameters(), self.grad_clip_value)
            self.critic_optim.step()

            self.logger['critic_loss'].append(
                                    critic_loss.cpu().detach().numpy())

    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # The discounted reward so far
            # Iterate through all rewards in the episode.
            # We go backwards for smoother calculation of each
            # discounted return
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

            # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def get_action(self, obs, eps=0., epsilon_greedy=False,
                    deterministic=False):
        # Query the actor network for a mean action
        conv_input, mask = self.actor_utils.create_conv_input([obs])
        conv_input = torch.tensor(conv_input)
        conv_input = conv_input.reshape(1, 1, -1)
        mask = torch.tensor(mask)
        # Move to cuda if available
        conv_input = conv_input.to(self.device)
        mask = mask.to(self.device)
        logits = self.actor(conv_input, mask).squeeze()
        dist = Categorical(logits=logits)
        action = None

        if epsilon_greedy:
            if random.random() > eps:
                action = torch.argmax(logits)
            else:
                action = random.choice(torch.where(
                    logits.squeeze() > -torch.inf)[-1])
        else:
            # Sample an action from the distribution
            #if self.t >= 0 and self.t <= 100000:
            #    logits = self.noise.get_action(logits, self.t)
            #    self.t += 1
            #    dist = Categorical(logits=logits)
            #    action = dist.sample()
            #elif self.t >= 25000*15:
            #    print('\nRestarting Exploration\n')
            #    print(f'logits={logits}')
            #    self.t = 0
            #    self.noise.reset()
            #    action = dist.sample()
            #else:
            #    self.t += 1
            #    action = dist.sample()
            action = dist.sample()

        if deterministic:
            action = torch.argmax(logits)

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log probability
        return action.cpu().detach().numpy(), log_prob.cpu().detach()

    def evaluate(self, batch_obs, batch_acts):
        # Query critic network for a value V for each batch_obs.
        # Shape of V should be same as batch_rtgs
        #values = self.critic(batch_obs, batch_acts).flatten()
        critic_input = self.critic_utils.create_input(batch_obs)
        critic_input = torch.tensor(critic_input)
        critic_input = critic_input.reshape(
                            len(batch_obs), len(batch_obs[0]), -1)
        # Move to cuda if available
        critic_input = critic_input.to(self.device)
        values = self.critic(critic_input).flatten()
        # Calculate the log probabilities of batch actions
        # using most recent actor network.
        # This segment of code is similar to that in get_action()
        conv_input, mask = self.actor_utils.create_conv_input(batch_obs)
        conv_input = torch.tensor(conv_input)
        conv_input = conv_input.reshape(len(batch_obs), len(batch_obs[0]), -1)
        mask = torch.tensor(mask)
        # Move to cuda if available
        conv_input = conv_input.to(self.device)
        mask = mask.to(self.device)
        logits = self.actor(conv_input, mask).squeeze()
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return values, log_probs.flatten()

    def _init_hyperparameters(self, hyperparameters):
        self.n_updates_per_iteration = hyperparameters['updates_per_iteration']\
                if 'updates_per_iteration' in hyperparameters else 5
        self.actor_lr = hyperparameters['actor_lr']\
                if 'actor_lr' in hyperparameters else 0.005
        self.critic_lr = hyperparameters['critic_lr']\
                if 'critic_lr' in hyperparameters else 0.001
        self.gamma = hyperparameters['gamma']\
                if 'gamma' in hyperparameters else 0.95
        self.clip = hyperparameters['clip']\
                if 'clip' in hyperparameters else 0.2
        self.grad_clip_value = hyperparameters['grad_clip_value']\
                if 'grad_clip_value' in hyperparameters else 100

class OUNoise(object):
    '''Class to implement OU Process'''
    def __init__(self, action_dim, mu=0.0, theta=0.15, max_sigma=2.5,
            min_sigma=0.05, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_dim
        self.reset()

    def reset(self):
        self.state = torch.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * torch.randn(
                self.action_dim
                )
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
                1.0, t / self.decay_period
                )
        # print(ou_state)
        return action + ou_state
