
import gym
import matplotlib
import random
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import namedtuple, deque
from itertools import count


env = gym.make("MountainCar-v0",render_mode="human")

device ="cpu"

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

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

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def memoryLen(self):
        return len(self.memory)
    

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4


# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)
print("num of actions",n_actions)
print("num of observations",n_observations)

policy_net = network(n_observations, n_actions).to(device)
target_net = network(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayBuffer(10000)


steps_done = 0


def selectAction(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []
losses = []
expected_values = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
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


            
def learn():
    if steps_done < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    next_state_betch = torch.cat(batch.next_state)
    reward_batch = torch.cat(batch.reward)
    done_batch = torch.cat(batch.done)

    state_action_values = policy_net(state_batch).gather(1, action_batch).view(1,BATCH_SIZE)
    target_action_values = target_net(next_state_betch).max(1)[0].view(1,BATCH_SIZE)
    # Compute the expected Q values
    expected_state_action_values = (target_action_values * GAMMA) * (1 - done_batch) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)
    
    #store expected values and loss of each step
    losses.append(loss.item())
    expected_values.extend(expected_state_action_values.detach().numpy())
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    
for episode in range(300):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0

    for t in count():
        env.render()
        action = selectAction(state)
        observation, reward, done, _ = env.step(action.item())[:4]
        reward = torch.tensor([reward], device=device)
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        done = torch.tensor(done, dtype=torch.float32, device=device).unsqueeze(0)

        memory.push(state, action, next_state, reward, done)
        state = next_state
        
        learn()
        
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for param in policy_net_state_dict:
            target_net_state_dict[param] = policy_net_state_dict[param]*TAU + target_net_state_dict[param]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        
        if done:
            episode_durations.append(t+1)
            plot_durations()
            break
    print('i_episode:', episode)
    print('learn_steps:', steps_done)

all_total_reward = episode_durations
print("Done")


plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.plot(torch.tensor(all_total_reward, dtype=torch.float))
plt.title('Returns for Cartpole')
plt.xlabel('Episodes')
plt.ylabel('returns')

plt.subplot(1, 3, 2)
plt.plot(losses)
plt.title("Training Loss Over Time")
plt.xlabel("Learn_steps")
plt.ylabel("Loss")

plt.subplot(1, 3, 3)
plt.plot(expected_values)
plt.title("Expected Values Over Time")
plt.xlabel("learn_steps")
plt.ylabel("Expected Value")
plt.savefig('/Users/yuyanlin/Desktop/AdaptiveTrafficLight/CartpoleDQN.png')

plt.show()

print("complete")

    