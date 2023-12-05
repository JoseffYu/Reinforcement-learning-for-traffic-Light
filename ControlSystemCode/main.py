import subprocess
from SUMOenv import SumoEnv
from DQN import DQN, ReplayBuffer
import matplotlib.pyplot as plt
import threading

device = "cpu"
sumo_cfg = "/Users/yuyanlin/Desktop/AdaptiveTrafficLight/Simulator/RunSimulator.sumocfg"

all_total_reward = []

def main():
    
    
    env = SumoEnv(sumo_cfg_file=sumo_cfg,
                  simulation_time=50
                  )
    simulation_time = 50
    input_dim = env.observation_space().shape[0]
    output_dim = env.action_space().n
    agent = DQN(env = env,mode="train",input_dim=input_dim,output_dim=output_dim,gamma=0.95,replay_size=20000,batch_size=32,eps_start=0.95,eps_end=0.05,eps_decay=30000,LR=1e-4)
    replay_buffer = ReplayBuffer(agent.replay_size)

    for episode in range(151):
        total_reward = 0
        initial_state = env.reset()
        
        env.train_state = initial_state
        done = False
        invalid_action = False

        while not done:
            state = env.computeState()
            action = agent.selectAction(state, agent.learn_step_counter, invalid_action)
            next_state, reward, done, info = env.step(env.tl_id[0],action)
            total_reward += reward
            #total_reward = env.traffic_light.total_reward
            if info['do_action'] is None:
                invalid_action = True
                continue
            invalid_action = False
            replay_buffer.push(state, next_state, reward, info['do_action'])
            agent.memory = replay_buffer
            agent.learn()
            #print("each step reward", reward)
        all_total_reward.append(total_reward)
        if episode % 5 ==0:
            #each_episode_return = []
            #each_episode_return.append(all_total_reward[0])
            #for i in range (1,len(all_total_reward)):
                #each_episode_return.append(all_total_reward[i]-all_total_reward[i-1])
            print(all_total_reward)
        
        env.close()
        
        print('i_episode:', episode)
        print('learn_steps:', agent.learn_step_counter)
    plt.plot(all_total_reward)
    plt.title('Returns for Cartpole')
    plt.xlabel('episode')
    plt.ylabel('returns')
    plt.savefig('/Users/yuyanlin/Desktop')

print(main())
