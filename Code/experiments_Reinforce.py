#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Using code structure from
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
import time
from REINFORCE import ReinforceAgent
import gym
from helper import LearningCurvePlot, smooth


def average_over_repetitions(n_states, n_actions, n_hidden, lr, gamma,
                 n_episodes, n_repetitions, smoothing_window, plot, entropy_coef,
                 experiment):

    train_results = np.empty([n_repetitions, n_episodes])
    now = time.time()
    env = gym.make('LunarLander-v2')
    if plot:
        env.render(mode='human')

    for rep in range(n_repetitions):
        print("Experiment {}: {}/{}" .format(experiment, rep+1, n_repetitions))

        agent = ReinforceAgent(n_states, n_actions, n_hidden, plot, lr, gamma, n_episodes, entropy_coef)
        # train the agent
        episode_rewards = agent.reinforce(env)
        train_results[rep] = episode_rewards

    print("the experiment took {} minutes".format((time.time() - now) / 60))
    learning_curve_train = np.mean(train_results, axis=0)  # average over repetitions
    learning_curve_train = smooth(learning_curve_train, smoothing_window)  # additional smoothing
    return learning_curve_train


def experiment():
    n_repetitions = 5
    smoothing_window = 250
    n_episodes = 5000
    n_states = 8
    n_actions = 4
    n_hidden = 64
    lr = 0.001
    gamma = 0.99
    plot = False
    entropy_coef = 0.01


    ''' Experiment 1: Reward per episode of REINFORCE '''
    experiment = 1
    Plot = LearningCurvePlot(title='Reward per Episode with REINFORCE')

    # Plot rewards per episode during training
    learning_curve_train = average_over_repetitions(n_states, n_actions, n_hidden, lr, gamma,
                 n_episodes, n_repetitions, smoothing_window, plot, entropy_coef, experiment)
    Plot.add_curve(learning_curve_train, label="Training")

    Plot.save('reinforce_experiment1.png')
    print("Plot saved at 'reinforce_experiment1.png'")


    ''' Experiment 2: Plot with different entropy regularization coefficients '''
    Plot = LearningCurvePlot(title='REINFORCE with different entropy regularization strengths')

    entropy_coef = 0.001
    learning_curve = average_over_repetitions(n_states, n_actions, n_hidden, lr, gamma,
                 n_episodes, n_repetitions, smoothing_window, plot, entropy_coef, experiment)
    Plot.add_curve(learning_curve, label="entropy coeff = 0")

    entropy_coef = 0.01
    learning_curve = average_over_repetitions(n_states, n_actions, n_hidden, lr, gamma,
                 n_episodes, n_repetitions, smoothing_window, plot, entropy_coef, experiment)
    Plot.add_curve(learning_curve, label="entropy coeff = 0.01")

    entropy_coef = 0.1
    learning_curve = average_over_repetitions(n_states, n_actions, n_hidden, lr, gamma,
                 n_episodes, n_repetitions, smoothing_window, plot, entropy_coef, experiment)
    Plot.add_curve(learning_curve, label="entropy coeff = 0.1")

    entropy_coef = 1.0
    learning_curve = average_over_repetitions(n_states, n_actions, n_hidden, lr, gamma,
                 n_episodes, n_repetitions, smoothing_window, plot, entropy_coef, experiment)
    Plot.add_curve(learning_curve, label="entropy coeff = 1.0")

    Plot.save('reinforce_experiment2.png')
    print("Plot saved at 'reinforce_experiment2.png'")

if __name__ == '__main__':
    experiment()
