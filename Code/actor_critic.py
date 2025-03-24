import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np
import gym
from helper import smooth, LearningCurvePlot
from tqdm import tqdm


class Actor(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_states, n_hidden),
            nn.Dropout(p=0.15),
            nn.PReLU(),
            nn.Linear(n_hidden, n_actions),

            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


# Define the Critic neural network class
class Critic(nn.Module):
    def __init__(self, n_states, n_hidden):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_states, n_hidden),  # Linear layer
            nn.Dropout(p=0.15),  # Dropout layer is the trick in this architecture
            nn.PReLU(),  # PRELU just slightly better than RELU

            nn.Linear(n_hidden, 1)  # Output layer
        )

    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, actor, critic, lr, n_episodes, gamma, use_bootstrap, use_baseline, n_steps, entropy_coeff, PPO,
                 PPO_STEPS):
        super().__init__()

        self.actor = actor
        self.critic = critic
        self.lr = lr
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.optimizer_actor = optim.Adam(actor.parameters(), lr=self.lr)
        self.optimizer_critic = optim.Adam(critic.parameters(), lr=self.lr)
        self.use_bootstrap = use_bootstrap
        self.use_baseline = use_baseline
        self.n_steps = n_steps
        self.entropy_coeff = entropy_coeff
        self.epsilon = 0.25
        self.PPO_STEPS = PPO_STEPS
        self.use_PPO = PPO

    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)

        return action_pred, value_pred

    def PPO_execution(self, states, actions, log_prob_actions, advantages, returns):
        states = torch.cat(states)
        actions = torch.cat(actions)
        log_prob_actions = torch.cat(log_prob_actions).detach()
        advantages = advantages.detach()

        for _ in range(self.PPO_STEPS):
            action_pred = self.actor(states)
            value_pred = self.critic(states).squeeze(-1)

            # Calculate the ratio term for PPO
            dist = distributions.Categorical(action_pred)
            new_log_prob_actions = dist.log_prob(actions)
            policy_ratio = (new_log_prob_actions - log_prob_actions).exp()

            # Calculate both clipped and unclipped objective
            policy_loss_1 = policy_ratio * advantages
            policy_loss_2 = torch.clamp(policy_ratio, min=1.0 - self.epsilon, max=1.0 + self.epsilon) * advantages

            # Compute the losses
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).sum()
            value_loss = F.smooth_l1_loss(returns, value_pred).sum()

            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            self.optimizer_actor.step()
            self.optimizer_critic.step()

    def bootstrap(self, rewards, values):
        if self.use_bootstrap:
            target = torch.zeros(len(rewards), dtype=torch.float)
            discount_factors = torch.tensor([pow(self.gamma, i) for i in range(self.n_steps)])
            r = torch.tensor(rewards)
            for t in range(len(target)):
                n = min(self.n_steps, len(target) - t)
                if t + n < len(target):
                    target[t] = torch.sum(r[t:t + n] * discount_factors[:n]) + discount_factors[n - 1] * values[
                        t + n]
                else:
                    target[t] = torch.sum(r[t:] * discount_factors[:n]) + discount_factors[-1] * values[-1]
            returns = torch.tensor(target)


        else:
            discounted_rewards = torch.zeros(len(rewards))
            R = 0
            for i, reward in enumerate(reversed(rewards)):
                R = reward + self.gamma * R
                discounted_rewards[i] = R
            discounted_rewards = discounted_rewards.flip(dims=(0,))
            returns = discounted_rewards
        return returns


    def train(self, env):
        print(self.use_bootstrap, self.use_baseline)
        train_rewards = []
        mean_rewards = []
        for episode in (pbar := tqdm(range(0, self.n_episodes))):
            ''' 1. Reset environment and initialize episode reward to 0 '''
            log_prob_actions = []
            values = []
            rewards = []
            states = []  # Create a list to store states
            done = False
            episode_reward = 0
            state = env.reset()
            states.append(state)
            states_PPO = []
            entropy = 0
            actions_PPO = []
            log_prob_actions_PPO = []

            while not done:

                ''' 2. Select an action according to current policy '''
                state = torch.FloatTensor(state).unsqueeze(0)
                if self.use_PPO:
                    states_PPO.append(state)
                action_pred = self.actor(state)
                value_pred = self.critic(state)

                # Action chosen based on the actor’s output probabilities
                #action_prob = F.softmax(action_pred, dim=-1)
                dist = distributions.Categorical(action_pred)
                action = dist.sample()
                log_prob_action = dist.log_prob(action)

                entropy += -(action_pred * torch.log(action_pred)).sum()

                ''' 3. Take the action and get observation and reward in return '''
                state, reward, done, _ = env.step(action.item())
                states.append(state)
                log_prob_actions.append(log_prob_action)
                values.append(value_pred)
                rewards.append(reward)
                episode_reward += reward

                if self.use_PPO:
                    actions_PPO.append(action)
                    log_prob_actions_PPO.append(log_prob_action)

            log_prob_actions = torch.cat(log_prob_actions)
            values = torch.cat(values).squeeze(-1)


            ''' 4. Bootstrap '''
            returns = self.bootstrap(rewards, values)

            ''' 5. Baseline: compute advantage function'''
            if self.use_baseline:
                advantages = returns - values
            else:
                advantages = returns

            if self.use_PPO:
                self.PPO_execution(states_PPO, actions_PPO, log_prob_actions_PPO, advantages, returns)

            else:
                ''' 6. Compute actor and critic losses based on advantage function '''
                advantages = advantages.detach()
                returns = returns.detach()
                policy_loss = - (advantages * log_prob_actions).sum() - self.entropy_coeff * entropy
                value_loss = ((values-returns)**2).sum()

                ''' 7. Compute the gradients and update actor and critic networks '''
                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                policy_loss.backward()
                value_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)

                self.optimizer_actor.step()
                self.optimizer_critic.step()

            train_rewards.append(episode_reward)

            if episode % 10 == 0:
                pbar.set_description(
                    f"Episode: {episode},  Episode Reward: {episode_reward}, "
                    f"Average score {np.mean(train_rewards)} ")
        return train_rewards

