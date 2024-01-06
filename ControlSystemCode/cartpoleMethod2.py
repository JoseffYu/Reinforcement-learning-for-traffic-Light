
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
                        ('state', 'action', 'next_state', 'reward'))

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


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
                 TAU,
                 LR):
        self.mode = mode
        self.n_actions = output_dim  #size of action space
        self.gamma = gamma
        self.TAU = TAU
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.lr = LR
        self.episode_durations = []
        self.memory = None
        self.env = env

        self.policy_net = network(input_dim, output_dim).to(device)
        self.target_net = network(input_dim, output_dim).to(device)
        self.target_net_state_dict = None
        self.policy_net_state_dict = None
        self.replay_size = replay_size

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.loss_func = nn.MSELoss()
        self.losses = []
        self.expected_values = []
        
        self.learn_step_counter = 0  # for target updating


    def selectAction(self, state, steps_done):
        #original_state = state
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * steps_done / self.eps_decay)
        self.learn_step_counter += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([random.randrange(self.env.action_space.n)]).unsqueeze(dim=1)


    def plot_durations(self,show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
    

    def learn(self):
        if self.learn_step_counter < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
    
        # Compute Huber loss
        loss = self.loss_func(state_action_values, expected_state_action_values)
        
        self.losses.append(loss.item())
        self.expected_values.extend(expected_state_action_values.detach().numpy())
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        self.learn_step_counter += 1