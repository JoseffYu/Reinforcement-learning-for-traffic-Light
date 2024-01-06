
import matplotlib
import random
import math
import networkx
import torch
import numpy as np
from datetime import datetime
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import namedtuple, deque

device ="cpu"

Transition = namedtuple('Transition',
                        ('state', 'next_state', 'reward', 'action'))


class network(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(network, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)



class ReplayBuffer(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.memory_len = 0

    def push(self, *args):
        self.memory.append(Transition(*args))
        self.memory_len += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def memoryLen(self):
        return self.memory_len


   
class DQN:
    def __init__(self,
                 env,
                 mode,
                 input_dim,
                 output_dim,
                 gamma,
                 replay_size,
                 batch_size,
                 eps_start: float,
                 eps_end: float,
                 eps_decay: int,
                 LR):
        self.mode = mode
        self.n_actions = output_dim  #size of action space
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.lr = LR
        self.episode_durations = []
        self.memory = None
        self.env = env

        self.eval_net = network(input_dim, output_dim).to(device)
        self.target_net = network(input_dim, output_dim).to(device)
        self.replay_size = replay_size
        self.losses = []
        self.expected_values = []

        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR, amsgrad=True)
        self.loss_func = nn.MSELoss()
        
        self.learn_step_counter = 0  # for target updating


    def selectAction(self, state, steps_done, invalid_action):
        #original_state = state
        state = torch.from_numpy(state)
        if self.mode == 'train':
            sample = random.random()
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * steps_done / self.eps_decay)
            self.learn_step_counter += 1
            if sample < eps_threshold:
                with torch.no_grad():                    
                    _, sorted_indices = torch.sort(self.eval_net(state), descending=True)
                    if invalid_action:
                        return sorted_indices[1]
                    else:
                        return sorted_indices[0]
            else:
                return self.env.action_space().sample()



    def learn(self):
        if self.learn_step_counter < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat([torch.tensor(batch.state)])
        action_batch = torch.cat([torch.tensor(batch.action)]).view(self.batch_size, 1)
        reward_batch = torch.cat([torch.tensor(batch.reward)]).view(self.batch_size, 1)
        next_state_batch = torch.cat([torch.tensor(batch.next_state)])
        state_action_values = self.eval_net(state_batch).gather(1, action_batch)
        
        
        with torch.no_grad():
            argmax_action = self.target_net(next_state_batch).max(1)[1].view(self.batch_size, 1)
        expected_state_action_values = reward_batch + self.gamma * self.target_net(next_state_batch).gather(1, argmax_action)  # Compute the expected Q values

        # Compute Huber loss
        loss = self.loss_func(state_action_values, expected_state_action_values)
        self.losses.append(loss.item())
        self.expected_values.extend(expected_state_action_values.detach().numpy())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.eval_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.learn_step_counter += 1
        
        if self.learn_step_counter%10==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())