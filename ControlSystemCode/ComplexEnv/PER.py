
import matplotlib
import random
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import namedtuple, deque

device ="cpu"

Transition = namedtuple('Transition',
                        ('state', 'next_state', 'reward', 'action','done'))

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




class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class ReplayBuffer_Per(object):
    # stored as ( s, a, r, s_ ) in SumTree
    def __init__(self, capacity=1000, a=0.6, e=0.01):
        self.tree = SumTree(capacity)
        self.memory_size = capacity
        self.prio_max = 0.1
        self.a = a
        self.e = e

    def push(self, *args):
        data = Transition(*args)
        p = (np.abs(self.prio_max) + self.e) ** self.a  # proportional priority
        self.tree.add(p, data)

    def sample(self, batch_size):
        idxs = []
        segment = self.tree.total() / batch_size
        sample_datas = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)

            sample_datas.append(data)
            idxs.append(idx)
        return sample_datas

    def update(self, idxs, errors):
        self.prio_max = max(self.prio_max, max(np.abs(errors)))
        for i, idx in enumerate(idxs):
            p = (np.abs(errors[i]) + self.e) ** self.a
            self.tree.update(idx, p)

    def size(self):
        return self.tree.n_entries



   
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
                 LR,
                 TAU):
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
        self.TAU = TAU

        self.policy_net = network(input_dim, output_dim).to(device)
        self.target_net = network(input_dim, output_dim).to(device)
        self.target_net_state_dict = self.target_net.state_dict()
        self.policy_net_state_dict = self.policy_net.state_dict()
        self.replay_size = replay_size
        self.losses = []
        self.expected_values = []

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.loss_func = nn.MSELoss()
        
        self.learn_step_counter = 0  # for target updating


    def selectAction(self, state, steps_done, invalid_action):
        #original_state = state
        state = torch.from_numpy(state)
        if self.mode == 'train':
            sample = random.random()
            #eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * steps_done / self.eps_decay)
            eps_threshold = 0.95
            self.learn_step_counter += 1
            if sample < eps_threshold:
                with torch.no_grad():
                    _, sorted_indices = torch.sort(self.policy_net(state), descending=True)
                    if invalid_action:
                        return sorted_indices[1]
                    else:
                        return sorted_indices[0]
            else:
                return self.env.action_space().sample()
            

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
        
        state_batch = torch.cat([torch.tensor(batch.state)])
        action_batch = torch.cat([torch.tensor(batch.action)]).view(self.batch_size,1)
        reward_batch = torch.cat([torch.tensor(batch.reward)]).view(self.batch_size,1)
        next_state_batch = torch.cat([torch.tensor(batch.next_state)])
        
        print(1)
        state_action_values = self.policy_net(state_batch).gather(1,action_batch).view(self.batch_size,1)
        target_action_values = self.target_net(next_state_batch).max(1)[0].view(self.batch_size,1)
        expected_state_action_values = reward_batch + self.gamma * target_action_values  # Compute the expected Q values

        # Compute Huber loss
        loss = self.loss_func(state_action_values, expected_state_action_values)
        #store expected values and loss of each step
        self.losses.append(loss.item())
        self.expected_values.extend(expected_state_action_values.detach().numpy())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
