import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import argparse
from functools import reduce
import torch.nn.functional as F
from tqdm import tqdm
from helper import LearningCurvePlot, smooth
import time


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x)
        x = torch.relu(self.fc1(x))
        x = nn.functional.softmax(self.fc2(x), dim=-1)
        return x


class ReinforceAgent():
    def __init__(self, n_states, n_actions, n_hidden, plot, lr, gamma, n_episodes, entropy_coef):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.plot = plot
        self.lr = lr
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.entropy_coef = entropy_coef
        self.do_gradient_analysis = False
        self.eval_interval = 50


        self.policy_net = PolicyNetwork(self.n_states, self.n_actions, self.n_hidden)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)


    def get_action(self, state):
        state = torch.tensor(state).float()
        action_probs = self.policy_net(state)

        action = np.random.choice(self.n_actions, p=action_probs.detach().numpy())
        return action, action_probs

    def compute_discounted_rewards(self, rewards_episode):
        discounted_rewards = torch.zeros(len(rewards_episode))
        R = 0
        for i, reward in enumerate(reversed(rewards_episode)):
            R = reward + self.gamma * R
            discounted_rewards[i] = R
        discounted_rewards = discounted_rewards.flip(dims=(0,))
        return discounted_rewards


    def gradient_analysis(self, i, fc1_weights, fc1_bias, fc2_weights, fc2_bias):
        gradient_analysis_interval = 50
        if i % gradient_analysis_interval == 0:
            for name, param in self.policy_net.named_parameters():
                average_gradient_magnitude = param.grad.abs().mean()  # Compute average gradient magnitude
                # print(f"Parameter: {name}, Average Gradient Magnitude: {average_gradient_magnitude}")
                if name == "fc1.weight":
                    fc1_weights.append(average_gradient_magnitude)
                elif name == "fc1.bias":
                    fc1_bias.append(average_gradient_magnitude)
                elif name == "fc2.weight":
                    fc2_weights.append(average_gradient_magnitude)
                elif name == "fc2.bias":
                    fc2_bias.append(average_gradient_magnitude)

    def evaluate(self, env):
        state = env.reset()
        done = False

        test_returns = []

        while not done:
            action_probs = self.policy_net(torch.tensor(state).float())

            action_space = env.action_space.n
            action = np.random.choice(action_space, p=action_probs.detach().numpy())

            next_state, reward, done, _ = env.step(action)
            state = next_state
            test_returns.append(reward)

        return test_returns

    def reinforce(self, env):
        rewards = []
        fc1_weights = []
        fc1_bias = []
        fc2_weights = []
        fc2_bias = []

        eval_returns = []
        eval_episodes = []

        for ep in (pbar := tqdm(range(0, self.n_episodes))):
            ''' 1. Reset the environment and get the initial state '''
            state = env.reset()
            log_probs = []
            rewards_episode = []
            done = False
            entropy = 0

            while not done:
                ''' 2. Sample an action based on the computed action probabilities '''
                action, action_probs = self.get_action(state)

                log_prob = torch.log(action_probs[action])
                log_probs.append(log_prob)

                entropy += -(action_probs * torch.log(action_probs)).sum()  # Update entropy

                ''' 3.  Take the action and observe the next state and reward '''
                state, reward, done, _ = env.step(action)
                rewards_episode.append(reward)

            rewards_episode = torch.tensor(rewards_episode)
            log_probs = torch.stack(log_probs)

            ''' 4. Compute the discounted rewards '''
            discounted_rewards = self.compute_discounted_rewards(rewards_episode)

            ''' 5. Compute the policy loss  and apply entropy regularization'''
            policy_loss = torch.sum(-log_probs * discounted_rewards) - self.entropy_coef * entropy

            ''' 6. Update the policy network '''
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            # Gradient analysis
            if self.do_gradient_analysis:
                self.gradient_analysis(ep, fc1_weights, fc1_bias, fc2_weights, fc2_bias)

            # Store the total rewards for the current episode
            rewards.append(sum(rewards_episode).item())

            if ep % 10 == 0:
                pbar.set_description(
                    f"Episode: {ep},  Episode Reward: {sum(rewards_episode).item()}, Average score {np.mean(rewards)} ")

            # Evaluate policy every eval_interval steps
            # if ep % self.eval_interval == 0:
            #     mean_return = self.evaluate(env)
            #     eval_returns.append(sum(mean_return))
            #     eval_episodes.append(ep)

        if self.do_gradient_analysis:
            print("fc1 weights: ", fc1_weights)
            print("\n")
            print("fc2 weights: ", fc2_weights)
            print("\n")
            print("fc1 bias: ", fc1_bias)
            print("\n")
            print("fc2 bias: ", fc2_bias)
            print("\n")
        return rewards


