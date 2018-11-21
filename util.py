import gym
import numpy as np
import matplotlib.pyplot as plt

def plotter(env_name, num_episodes, rewards_list, ylim):
    '''
    used to plot avg scores.
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
    used to plot raw scores
    :param scores:
    :return:
    '''
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Episode Rewards')
    plt.xlabel('Number of Episodes')
    plt.title("Raw Scores Over Time")
    plt.savefig("RawScore.png")
    plt.close()
