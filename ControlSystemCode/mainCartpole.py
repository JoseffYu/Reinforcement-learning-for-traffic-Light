import subprocess
import gym
from SUMOenv import SumoEnv
from cartpoleDQN import DQN, ReplayBuffer
import matplotlib.pyplot as plt
import numpy as np
import torch
import random

device = "cpu"
sumo_cfg = "/Users/yuyanlin/Desktop/AdaptiveTrafficLight/Simulator/RunSimulator.sumocfg"

all_total_reward = []

def main():
    
    env = gym.make("CartPole-v0")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQN(env = env,mode="train",input_dim=state_size,output_dim=action_size,gamma=0.95,replay_size=20000,batch_size=32,eps_start=0.95,eps_end=0.05,eps_decay=30000,LR=1e-4)
    replay_buffer = ReplayBuffer(agent.replay_size)

    for episode in range(151):
        total_reward = 0
        state = env.reset()[0]
        
        done = False
        invalid_action = False

        while not done:
            state = torch.FloatTensor(state).to(device)
            action = agent.selectAction(state, agent.learn_step_counter, invalid_action)
            next_state, reward, done, _ = env.step(action)[:4]
            total_reward += reward
            #total_reward = env.traffic_light.total_reward
            #if info['do_action'] is None:
                #invalid_action = True
                #continue
            invalid_action = False
            replay_buffer.push(state, next_state, reward, action)
            agent.memory = replay_buffer
            agent.learn()
                
        all_total_reward.append(total_reward)
        if episode % 5 ==0:
            print(all_total_reward)
        
        env.close()
        
        print('i_episode:', episode)
        print('learn_steps:', agent.learn_step_counter)
    plt.plot(all_total_reward)
    plt.title('Returns for Cartpole')
    plt.xlabel('episode')
    plt.ylabel('returns')
    plt.savefig('/Users/yuyanlin/Desktop/CS342')

print(main())
