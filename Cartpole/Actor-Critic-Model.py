from itertools import count
import torch
import gym
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cpu")
env = gym.make("CartPole-v1")

state_size = env.observation_space.shape[0]
state_ex = env.observation_space
action_size = env.action_space.n
lr = 0.0001

class Actor(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = torch.nn.Linear(self.state_size, 128)
        self.linear2 = torch.nn.Linear(128, 256)
        self.linear3 = torch.nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = torch.distributions.Categorical(F.softmax(output, dim=-1))
        return distribution


class Critic(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = torch.nn.Linear(self.state_size, 128)
        self.linear2 = torch.nn.Linear(128, 256)
        self.linear3 = torch.nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value
    
    def discounted_rewards(self, next_value, rewards, masks, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns

    def run_episode(critic, actor, n_iters):
        
        optimizerA = torch.optim.Adam(actor.parameters())
        optimizerC = torch.optim.Adam(critic.parameters())
        for iter in range(n_iters):
            state = env.reset()[0]
            print(env.reset())
            log_probs = []
            values = []
            rewards = []
            masks = []
            entropy = 0
            env.reset()

            for i in count():
                env.render()
                state = torch.FloatTensor(state).to(device)
                dist, value = actor.forward(state), critic.forward(state)
                action = dist.sample()
                next_state, reward, done, _ = env.step(action.cpu().numpy())[:4]

                log_prob = dist.log_prob(action).unsqueeze(0)
                entropy += dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
                masks.append(torch.tensor([1-done], dtype=torch.float, device=device))

                state = next_state

                if done:
                    print('Iteration: {}, Score: {}'.format(iter, i))
                    break
        
            next_state = torch.FloatTensor(next_state).to(device)
            next_value = critic(next_state)
            returns = critic.discounted_rewards(next_value, rewards, masks)

            log_probs = torch.cat(log_probs)
            returns = torch.cat(returns).detach()
            values = torch.cat(values)

            advantage = returns - values

            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()

            optimizerA.zero_grad()
            optimizerC.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            optimizerA.step()
            optimizerC.step()
            

actor = Actor(state_size, action_size).to(device)
critic = Critic(state_size, action_size).to(device)
critic.run_episode(actor, n_iters=1000)
print(state_size,action_size)
print(state_ex)
env.close()
