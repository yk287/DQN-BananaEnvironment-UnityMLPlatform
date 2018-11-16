import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F


class pytorch_DQNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, layer_1 = 64, layer_2 = 64):
        '''
        :param state_size:
        :param action_size:
        :param seed:
        :param layer_1:
        :param layer_2:
        '''

        super(pytorch_DQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        '''specify the network'''
        self.fully_connected1 = nn.Linear(state_size, layer_1)
        self.fully_connected2 = nn.Linear(layer_1, layer_2)
        self.fully_connected3 = nn.Linear(layer_2, action_size)


    def forward(self, state):
        '''
        overrides the function forward
        :param state:
        :return:
        '''

        temp = F.relu(self.fully_connected1(state))
        temp = F.relu(self.fully_connected2(temp))

        return self.fully_connected3(temp)
