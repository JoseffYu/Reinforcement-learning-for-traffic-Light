import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt2
import torch
import gym

# build env
env = gym.make("CartPole-v1")

num_input = 4
num_actions = 2
eps = 0.0001
alpha = 1e-4


model = torch.nn.Sequential(
    torch.nn.Linear(num_input,128,bias=False,dtype=torch.float32),
    torch.nn.ReLU(),
    torch.nn.Linear(128,num_actions,bias=False,dtype=torch.float32),
    torch.nn.Softmax(dim=1)
)

optimizer = torch.optim.Adam(model.parameters(),lr=0.01)


def runEpisode(max_steps_per_episode=10000, render=False):    
    
    states, actions, probs, rewards = [],[],[],[]
    state = env.reset()[0]
    
    for _ in range(max_steps_per_episode):
        if render:
            env.render()
        
        action_probs = model(torch.from_numpy(np.expand_dims(state,0)))[0]
        action = np.random.choice(num_actions, p=np.squeeze(action_probs.detach().numpy()))
        nstate, reward, done, info = env.step(action)[:4]
    
        if done:
            break    
        
        states.append(state)
        actions.append(action)
        probs.append(action_probs.detach().numpy())
        rewards.append(reward)
        state = nstate
        
    return np.vstack(states), np.vstack(actions), np.vstack(probs), np.vstack(rewards)


def discountRewards(reward,gamma=0.99, normalise=True):
    
    ret = []
    s = 0
    
    for r in reward[::1]:
        s = r + gamma*s
        ret.insert(0,s)
    
    if normalise:
        ret = (ret-np.mean(ret))/(np.std(ret)+eps)
    
    return ret


def trainOnBatch(x, y):
    
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
   
    optimizer.zero_grad()
    predictions = model(x)
    loss = -torch.mean(torch.log(predictions)*y)
    
    loss.backward()
    optimizer.step()
    
    return loss


history = []

for epoch in range(300):
    states, actions, probs, rewards = runEpisode()
    one_hot_actions = np.eye(2)[actions.T][0]
    gradients = one_hot_actions - probs
    dr = discountRewards(rewards)
    gradients *= dr
    target = alpha * np.vstack([gradients]) + probs
    trainOnBatch(states,target)
    history.append(np.sum(rewards))
    if epoch%100==0:
        print(f"{epoch} -> {np.sum(rewards)}")

plt2.plot(history)

s, a, p, r = runEpisode()


_ = runEpisode(render=True)

print(f"Total reward: {np.sum(r)}")