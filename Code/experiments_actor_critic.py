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
from actor_critic import ActorCritic, Actor, Critic
import gym
from helper import LearningCurvePlot, smooth


def average_over_repetitions_AC(actor, critic, lr, gamma,
                 n_episodes, n_repetitions, smoothing_window, plot, entropy_coeff, experiment,
                use_bootstrap, use_baseline, model, n_steps, PPO, PPO_STEPS):

    train_results = np.empty([n_repetitions, n_episodes])
    now = time.time()
    env = gym.make('LunarLander-v2')
    if plot:
        env.render(mode='human')

    for rep in range(n_repetitions):
        print("Experiment {}: {}/{}" .format(experiment, rep+1, n_repetitions))

        agent = ActorCritic(actor, critic, lr, n_episodes, gamma, use_bootstrap, use_baseline, n_steps, entropy_coeff,
                            PPO, PPO_STEPS)
        # train the agent
        episode_rewards = agent.train(env)

        train_results[rep] = episode_rewards
        np.save(f'M{model}episode_rewards_rep{rep+1}.npy', episode_rewards)

    print("the experiment took {} minutes".format((time.time() - now) / 60))
    learning_curve_train = np.mean(train_results, axis=0)  # average over repetitions
    learning_curve_train = smooth(learning_curve_train, smoothing_window)  # additional smoothing
    return learning_curve_train

def average_over_repetitions_REINFORCE(n_states, n_actions, n_hidden, lr, gamma,
                 n_episodes, n_repetitions, smoothing_window, plot, entropy_coef,
                 experiment):

    train_results = np.empty([n_repetitions, n_episodes])
    now = time.time()
    env = gym.make('LunarLander-v2')
    if plot:
        env.render(mode='human')

    for rep in range(n_repetitions):
        print("Experiment {}: {}/{}" .format(experiment, rep, n_repetitions))

        agent = ReinforceAgent(n_states, n_actions, n_hidden, plot, lr, gamma, n_episodes, entropy_coef)
        # train the agent
        episode_rewards = agent.reinforce(env)

        train_results[rep] = episode_rewards

    print("the experiment took {} minutes".format((time.time() - now) / 60))
    learning_curve_train = np.mean(train_results, axis=0)  # average over repetitions
    learning_curve_train = smooth(learning_curve_train, smoothing_window)  # additional smoothing
    return learning_curve_train


def experiment():

    n_repetitions = 1
    smoothing_window = 250
    n_episodes = 3000
    n_states = 8
    n_actions = 4
    n_hidden = 128
    lr = 0.001
    gamma = 0.99
    plot = False # To render the environment
    PPO = False # If True, executes the actor critic with PPO
    entropy_coef = 0.01
    n_steps = 50
    use_bootstrap = True
    use_baseline = True
    PPO_STEPS = 20

    actor = Actor(n_states, n_actions, n_hidden)
    critic = Critic(n_states, n_hidden)


    ''' Experiment 1: Comparison Actor Critic VS REINFORCE '''
    Plot = LearningCurvePlot(title='REINFORCE vs Actor Critics')

    # 1. REINFORCE
    learning_curve_train = average_over_repetitions_REINFORCE(n_states, n_actions, n_hidden, lr, gamma,
                 n_episodes, n_repetitions, smoothing_window, plot, entropy_coef, experiment)
    Plot.add_curve(learning_curve_train, label="REINFORCE")

    # 2. Actor Critic with bootstrap and baseline subtraction
    model = 'A'
    learning_curve_train = average_over_repetitions_AC(actor, critic, lr, gamma,
                 n_episodes, n_repetitions, smoothing_window, plot, entropy_coef, experiment,
                use_bootstrap, use_baseline, model, n_steps, PPO, PPO_STEPS)
    Plot.add_curve(learning_curve_train, label="AC")

    # 3. Actor critic with bootstrap and without baseline subtraction
    use_bootstrap = True
    use_baseline = False
    model = 'B'
    learning_curve_train = average_over_repetitions_AC(actor, critic, lr, gamma,
                 n_episodes, n_repetitions, smoothing_window, plot, entropy_coef, experiment,
                                                    use_bootstrap, use_baseline, model, n_steps, PPO, PPO_STEPS)
    Plot.add_curve(learning_curve_train, label="AC without baseline subtraction")

    # 4. Actor critic without bootstrap and with baseline subtraction
    use_bootstrap = False
    use_baseline = True
    model = 'C'
    learning_curve_train = average_over_repetitions_AC(actor, critic, lr, gamma,
                 n_episodes, n_repetitions, smoothing_window, plot, entropy_coef, experiment,
                                                    use_bootstrap, use_baseline, model, n_steps, PPO, PPO_STEPS)
    Plot.add_curve(learning_curve_train, label="AC without bootstrap")

    Plot.save('models_comparison.png')
    print("Plot saved at 'models_comparison.png'")



    ''' Experiment 2: n-steps values comparison '''
    use_baseline = True
    use_bootstrap = True

    Plot = LearningCurvePlot(title='Actor-Critic n-steps Comparison')

    # 1. 1-step AC
    n_steps = 1
    model = 'D'
    learning_curve = average_over_repetitions_AC(actor, critic, lr, gamma,
                 n_episodes, n_repetitions, smoothing_window, plot, entropy_coef, experiment,
                                                    use_bootstrap, use_baseline, model, n_steps, PPO, PPO_STEPS)
    Plot.add_curve(learning_curve, label="n-steps=1")

    # 2. 5-step AC
    n_steps = 5
    model = 'E'
    learning_curve = average_over_repetitions_AC(actor, critic, lr, gamma,
                 n_episodes, n_repetitions, smoothing_window, plot, entropy_coef, experiment,
                                                    use_bootstrap, use_baseline, model, n_steps, PPO, PPO_STEPS)
    Plot.add_curve(learning_curve, label="n-steps=5")

    # 3. 10-step AC
    n_steps = 10
    model = 'F'
    learning_curve = average_over_repetitions_AC(actor, critic, lr, gamma,
                 n_episodes, n_repetitions, smoothing_window, plot, entropy_coef, experiment,
                                                    use_bootstrap, use_baseline, model, n_steps, PPO, PPO_STEPS)
    Plot.add_curve(learning_curve, label="n-steps=10")

    # 4. 50-step AC
    n_steps = 50
    model = 'G'
    learning_curve = average_over_repetitions_AC(actor, critic, lr, gamma,
                 n_episodes, n_repetitions, smoothing_window, plot, entropy_coef, experiment,
                use_bootstrap, use_baseline, model, n_steps, PPO, PPO_STEPS)
    Plot.add_curve(learning_curve, label="n-steps=50")

    Plot.save('nsteps_comparison.png')
    print("Plot saved at 'nsteps_comparison.png'")




    ''' Experiment 3: entropy coefficient values comparison '''
    n_steps = 50
    Plot = LearningCurvePlot(title='Actor-Critic entropy strength Comparison')

    # 1. No entropy regularization
    entropy_coef = 0
    model = 'H'
    learning_curve = average_over_repetitions_AC(actor, critic, lr, gamma,
                 n_episodes, n_repetitions, smoothing_window, plot, entropy_coef, experiment,
                                                    use_bootstrap, use_baseline, model, n_steps, PPO, PPO_STEPS)
    Plot.add_curve(learning_curve, label="entropy coeff=0")

    # 2. Entropy strength of 0.01
    entropy_coef = 0.01
    model = 'I'
    learning_curve = average_over_repetitions_AC(actor, critic, lr, gamma,
                 n_episodes, n_repetitions, smoothing_window, plot, entropy_coef, experiment,
                                                    use_bootstrap, use_baseline, model, n_steps, PPO, PPO_STEPS)
    Plot.add_curve(learning_curve, label="entropy coeff=0.01")

    # 3. Entropy strength of 0.1
    entropy_coef = 0.1
    model = 'J'
    learning_curve = average_over_repetitions_AC(actor, critic, lr, gamma,
                 n_episodes, n_repetitions, smoothing_window, plot, entropy_coef, experiment,
                                                    use_bootstrap, use_baseline, model, n_steps, PPO, PPO_STEPS)
    Plot.add_curve(learning_curve, label="entropy coeff=0.1")

    # 4. Entropy strength of 1.0
    entropy_coef = 1.0
    model = 'K'
    learning_curve = average_over_repetitions_AC(actor, critic, lr, gamma,
                 n_episodes, n_repetitions, smoothing_window, plot, entropy_coef, experiment,
                use_bootstrap, use_baseline, model, n_steps, PPO, PPO_STEPS)
    Plot.add_curve(learning_curve, label="entropy coeff=1.0")

    Plot.save('entropy_comparison.png')
    print("Plot saved at 'entropy_comparison.png'")




    ''' Experiment 4: Bonus - PPO '''
    entropy_coef = 0.01
    Plot = LearningCurvePlot(title='Actor-Critic with and without PPO')

    # 1. No PPO
    model = 'L'
    learning_curve = average_over_repetitions_AC(actor, critic, lr, gamma,
                 n_episodes, n_repetitions, smoothing_window, plot, entropy_coef, experiment,
                                                    use_bootstrap, use_baseline, model, n_steps, PPO, PPO_STEPS)
    Plot.add_curve(learning_curve, label="Without PPO")

    # 2. With PPO
    PPO = True
    model = 'M'
    PPO_STEPS = 20
    learning_curve = average_over_repetitions_AC(actor, critic, lr, gamma,
                 n_episodes, n_repetitions, smoothing_window, plot, entropy_coef, experiment,
                                                    use_bootstrap, use_baseline, model, n_steps, PPO, PPO_STEPS)
    Plot.add_curve(learning_curve, label="PPO steps = 20")

    PPO = True
    model = 'N'
    PPO_STEPS = 50
    learning_curve = average_over_repetitions_AC(actor, critic, lr, gamma,
                 n_episodes, n_repetitions, smoothing_window, plot, entropy_coef, experiment,
                                                    use_bootstrap, use_baseline, model, n_steps, PPO, PPO_STEPS)
    Plot.add_curve(learning_curve, label="PPO steps = 50")
    Plot.save('ac_ppo.png')
    print("Plot saved at 'ac_ppo.png'")

if __name__ == '__main__':
    experiment()
