import gym
import numpy as np
import matplotlib.pyplot as plt

class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


def plotter(env_name, num_episodes, rewards_list, ylim):
    '''

    :param env_name:
    :param num_episodes:
    :param rewards_list:
    :param ylim:
    :return:
    '''
    x = np.arange(0, num_episodes)
    y = np.asarray(rewards_list)
    plt.plot(x, y)
    plt.ylim(top=ylim + 10)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Avg Rewards Last 100 Episodes")
    plt.title("Rewards Over Time For %s" %env_name)
    plt.savefig("progress.png")
    plt.close()

def raw_score_plotter(scores):
    '''

    :param scores:
    :return:
    '''
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Episode Rewards')
    plt.xlabel('Number of Episodes')
    plt.title("Raw Scores Over Time")
    plt.savefig("RawScore.png")
    plt.close()
