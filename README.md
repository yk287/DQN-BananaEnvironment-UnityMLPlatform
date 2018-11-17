
## An Implementation of DQN algorithm that solves Banana Collector Environment

![](banana.gif)

## Introduction

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Installation

#### Step 1: Clone the repo
The repo includes the Unity Environment for Linux OS

#### Step 2: Install Dependencies
Easiest way is to create an [anaconda](https://www.anaconda.com/download/) environment that contains all the required dependencies to run the project. Other than the modules that come with the Anaconda environment, **Pytorch** and **unityagents** are required. 

```
conda create --name BananaEnvironment python=3.6
source activate BananaEnvironment
conda install -y pytorch -c pytorch
pip install unityagents
```

## Training

To train the agent that learns how to solve the environment, simply run './main.py'. This will start the training process with the default hyperparameters given to the model. When the environment is solved, the script saves the model parameters and also outputs a couple of graphs that shows the rewards per episode, and the average rewards last 100 episodes.

Weights for the model that successfully achived an average score of 13+ over 100 episodes are included as **succesful_model.pth**
