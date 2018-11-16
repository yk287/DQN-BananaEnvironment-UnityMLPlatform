#main python code that trains an agent

import torch
import numpy as np
from collections import deque
from unityagents import UnityEnvironment

#used for plotting
from util import plotter, raw_score_plotter

#import DQN agent
from agent import Agent

env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")
env_name = 'Banana'

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of actions
action_size = brain.vector_action_space_size

# examine the state space
state = env_info.vector_observations[0]
state_size = len(state)

#instantiate an agent
agent = Agent(state_size, action_size,)

def DQN(num_episodes = 1500, threshold = 13, init_epsilon = 1.0, min_epsilon = 0.05, decay = 0.99):
    '''
    :param num_episodes:
    :param max_iteration:
    :param init_epsilon:
    :param min_epsilon:
    :param decay:
    :return:
    '''

    total_reward = []
    avg_score_last_100 = []
    total_reward_window = deque(maxlen=100)
    epsilon = init_epsilon
    PRINT_EVERY = 5

    for episodes in range(num_episodes):

        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations[0]
        rewards = 0
        done = 0

        while done == 0:

            action = agent.act(state, epsilon)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)

            state = next_state
            rewards += reward

            if done:
                break
        #a deque object used to hold values for the last 100 scores
        total_reward_window.append(rewards)

        #a list that holds the average score of the last 100 episodes
        avg_score_last_100.append(np.mean(total_reward_window))

        #total_reward holds all the rewards
        total_reward.append(rewards)

        #epsilon is decayed after every episode
        epsilon = max(min_epsilon, epsilon * decay)

        print('\rEpisode {}\tAverage Score: {:.3f}\tScore: {:.3f}'.format(episodes, avg_score_last_100[-1], rewards), end="")
        if episodes % PRINT_EVERY == 0:
            print('\rEpisode {}\tAverage Score: {:.3f}\tScore: {:.3f}'.format(episodes, avg_score_last_100[-1], rewards))

        if avg_score_last_100[-1] >= threshold:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}\tScore: {:3f}'.format(episodes - 100, avg_score_last_100[-1], rewards))
            torch.save(agent.local_model.state_dict(), 'successful_model.pth')
            break

    return total_reward, avg_score_last_100

num_episodes = 10000
threshold = 13

scores, avg_last_100 = DQN(num_episodes, threshold)

raw_score_plotter(scores)
plotter(env_name, len(scores), avg_last_100, threshold)
