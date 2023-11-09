
import SUMOenv

import random
import math
import networkx
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from collections import namedtuple

device ="cpu"

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

batch_size = 32
lr = 0.01
epsilon = 0.9
gamma= 0.9
discount= 100
capacity = 2000

n_actions = SUMOenv.action_space.n
n_states = SUMOenv.observation_space.shape[0]

class network:

    def __init__(self):
        super(network, self).__init__()

        self.lay1 = nn.Linear(n_states, 512)
        self.lay2 = nn.Linear(512, n_actions)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        actions_value = self.out(x)
        return actions_value

   
class DQN:
    def __init__(
        self,
    ):
        self = self


    def select_action(self, state, steps_done):
        return True


    def learn(self):
        return True
    