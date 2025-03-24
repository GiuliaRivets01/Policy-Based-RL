from actor_critic import ActorCritic, Actor, Critic
import numpy as np
import itertools
import gym
from joblib import parallel_backend
import time


def main():
    # Defining search space
    hpSearchSpace = {
        'lr': [0.001, 0.01, 0.1],
        'entropy_coeff': [0.01, 0.1],
        'gamma': [0.9, 0.99]
    }

    # Actor Critic with bootstrap and baseline subtraction
    env = gym.make('LunarLander-v2')
    bestParameters = grid_search(env, hpSearchSpace, True, True)
    print("Best parameters after hyperparameter tuning:")
    print(bestParameters)

    # Actor Critic with bootstrap and without baseline subtraction
    env = gym.make('LunarLander-v2')
    bestParameters = grid_search(env, hpSearchSpace, True, False)
    print("Best parameters after hyperparameter tuning:")
    print(bestParameters)

    # Actor Critic without bootstrap and with baseline subtraction
    env = gym.make('LunarLander-v2')
    bestParameters = grid_search(env, hpSearchSpace, False, True)
    print("Best parameters after hyperparameter tuning:")
    print(bestParameters)

    return


def grid_search(env, hpSearchSpace, use_bootstrap, use_baseline):
    bestAvgReward = float('-inf')
    bestParameters = None
    n_states = 8
    n_actions = 4
    n_hidden = 128

    with parallel_backend('multiprocessing', n_jobs=-1):
        # Iterate over each combination
        num_combinations = len(list(itertools.product(*hpSearchSpace.values())))
        print(num_combinations)
        i = 1
        for parametersCombination in itertools.product(*hpSearchSpace.values()):
            parametersDictionary = dict(zip(hpSearchSpace.keys(), parametersCombination))

            actor = Actor(n_states, n_actions, n_hidden)
            critic = Critic(n_states, n_hidden)

            agent = ActorCritic(actor=actor, critic=critic, n_episodes=3000, use_bootstrap=use_bootstrap,
                                use_baseline=use_baseline, n_steps=50, PPO=False, PPO_STEPS=20, **parametersDictionary)
            print("Running with the following hyperparameters combin {}/{}:".format(i, num_combinations))
            print(parametersDictionary)
            i += 1
            # Train the agent
            start = time.time()
            episode_rewards = agent.train(env)
            end = time.time() - start
            # Evaluate the performance of the agent and save results
            avg_episode_reward = np.mean(episode_rewards)
            print("It took {} minutes, result = {}".format(end / 60, avg_episode_reward))

            if avg_episode_reward > bestAvgReward:
                bestAvgReward = avg_episode_reward
                bestParameters = parametersDictionary
    return bestParameters


if __name__ == '__main__':
    main()


# Bootstrao = True, Baseline = rue
# result = -92.17721488152185 {'lr': 0.001, 'entropy_coeff': 0.01, 'gamma': 0.9}
# result = 20.741879748783628 {'lr': 0.001, 'entropy_coeff': 0.01, 'gamma': 0.99}
# result = -2.447191744793061 {'lr': 0.001, 'entropy_coeff': 0.1, 'gamma': 0.99}
# {'lr': 0.01, 'entropy_coeff': 0.1, 'gamma': 0.99} result = -16.744646151655402
# {'lr': 0.01, 'entropy_coeff': 0.1, 'gamma': 0.9}  result = -90.29560828240474
# {'lr': 0.01, 'entropy_coeff': 0.01, 'gamma': 0.99} result = 19.68290722514977
# {'lr': 0.1, 'entropy_coeff': 0.01, 'gamma': 0.99} result = -0.05124629328913094
# {'lr': 0.1, 'entropy_coeff': 0.1, 'gamma': 0.99} result = -34.15387084718818


# Bootstrap = True, Baseline = False
# {'lr': 0.001, 'entropy_coeff': 0.01, 'gamma': 0.9}  result = -101.08206281945141
# {'lr': 0.001, 'entropy_coeff': 0.01, 'gamma': 0.99} result = -267.0601482598834
# {'lr': 0.001, 'entropy_coeff': 0.1, 'gamma': 0.99} result = -337.4286776042949
# {'lr': 0.01, 'entropy_coeff': 0.1, 'gamma': 0.99}  result = -29.63829855921895
# {'lr': 0.01, 'entropy_coeff': 0.01, 'gamma': 0.99} result = -67.4586348696926
# {'lr': 0.1, 'entropy_coeff': 0.1, 'gamma': 0.99}  result = -345.70578881950155
# {'lr': 0.1, 'entropy_coeff': 0.01, 'gamma': 0.99} result = -184.01580579540152
# Learning rate = 0.01, entropy coeff = 0.1, gamma = 0.9  result = -86.94879819945595



# Bootstrap = False, Baseline = True
# {'lr': 0.001, 'entropy_coeff': 0.1, 'gamma': 0.99} result = 15.129852928881581
# {'lr': 0.001, 'entropy_coeff': 0.01, 'gamma': 0.99}  result = -13.68148914567
# {'lr': 0.01, 'entropy_coeff': 0.1, 'gamma': 0.99}  result = 11.497906001007136
# {'lr': 0.01, 'entropy_coeff': 0.01, 'gamma': 0.99} result = 16.916058840813815
# {'lr': 0.1, 'entropy_coeff': 0.01, 'gamma': 0.99} result = 33.513700050485745
# {'lr': 0.1, 'entropy_coeff': 0.1, 'gamma': 0.99} result = -9.188894738913174
# Learning rate = 0.1, entropy coeff = 0.01, gamma = 0.9  result = -88.61193373476951