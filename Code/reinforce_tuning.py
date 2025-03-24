from REINFORCE import ReinforceAgent
import numpy as np
import itertools
import gym
from joblib import parallel_backend


def main():
    # Defining search space
    hpSearchSpace = {
        'lr': [0.0001, 0.001, 0.01],
        'gamma': [0.9, 0.95, 0.99],
        'entropy_coef': [0.01, 0.1, 1.0]
    }


    env = gym.make('LunarLander-v2')

    # Generate all possible combinations of hyperparameters
    bestParameters = grid_search(env, hpSearchSpace)
    print("Best parameters after hyperparameter tuning:")
    print(bestParameters)

    return


def grid_search(env, hpSearchSpace):
    bestAvgReward = float('-inf')
    bestParameters = None

    with parallel_backend('multiprocessing', n_jobs=-1):
        # Iterate over each combination
        num_combinations = len(list(itertools.product(*hpSearchSpace.values())))
        print(num_combinations)
        i = 1
        for parametersCombination in itertools.product(*hpSearchSpace.values()):
            parametersDictionary = dict(zip(hpSearchSpace.keys(), parametersCombination))

            agent = ReinforceAgent(n_states=8, n_actions=4, n_hidden=64, plot=False, n_episodes=5000,
                             **parametersDictionary)

            print("Running with the following hyperparameters combin {}/{}:".format(i, num_combinations))
            print(parametersDictionary)
            i += 1
            # Train the agent
            episode_rewards = agent.reinforce(env)
            # Evaluate the performance of the agent and save results
            avg_episode_reward = np.mean(episode_rewards)
            print(avg_episode_reward)

            if avg_episode_reward > bestAvgReward:
                bestAvgReward = avg_episode_reward
                bestParameters = parametersDictionary
    return bestParameters


if __name__ == '__main__':
    main()