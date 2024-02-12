import subprocess
from SUMOenv import SumoEnv
from DqnChangeEpsilon import DQN, ReplayBuffer
#from DqnFixEpsilon import DQN, ReplayBuffer
#from DqnRandom import DQN,ReplayBuffer
#from PER import DQN, ReplayBuffer_Per
import matplotlib.pyplot as plt

device = "cpu"
sumo_cfg = "/Users/yuyanlin/Desktop/AdaptiveTrafficLight/Simulator/RunSimulator.sumocfg"

all_total_reward = []

def main():
    
    
    env = SumoEnv(sumo_cfg_file=sumo_cfg,
                  simulation_time=1000
                  )
    simulation_time = 50
    input_dim = env.observation_space().shape[0]
    output_dim = env.action_space().n
    agent = DQN(env = env,mode="train",input_dim=input_dim,output_dim=output_dim,gamma=0.95,replay_size=20000,batch_size=32,eps_start=0.95,eps_end=0.05,eps_decay=30000,LR=1e-4,TAU=0.1)
    #replay_buffer = ReplayBuffer_Per(agent.replay_size)
    replay_buffer = ReplayBuffer(agent.replay_size)

    for episode in range(100):
        total_reward = 0
        env.reset()
        state = env.computeState()
        done = False
        invalid_action = False

        while not done:
            action = agent.selectAction(state, agent.learn_step_counter, invalid_action)
            state,next_state, reward, done, info = env.step(env.tl_id[0],action)
            print("reward: ",reward)
            total_reward += reward
            #total_reward = env.traffic_light.total_reward
            if info['do_action'] is None:
                invalid_action = True
                continue
            invalid_action = False
            
            replay_buffer.push(state, next_state, reward, info['do_action'], done)
            agent.memory = replay_buffer
            state = next_state
            
            agent.learn()
            
            for param in agent.policy_net_state_dict:
                agent.target_net_state_dict[param] = agent.policy_net_state_dict[param]*agent.TAU + agent.target_net_state_dict[param]*(1-agent.TAU)
            agent.target_net.load_state_dict(agent.target_net_state_dict)
            #print("each step reward", reward)
        all_total_reward.append(total_reward)
        
        env.close()
        
        print('i_episode:', episode)
        print('learn_steps:', agent.learn_step_counter)
        
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(all_total_reward)
    plt.title('Returns for ATL(changing epsilon)')
    plt.xlabel('Episodes')
    plt.ylabel('returns')
    
    plt.subplot(1, 3, 2)
    plt.plot(agent.losses)
    plt.title("Training Loss Over Time")
    plt.xlabel("Learn_steps")
    plt.ylabel("Loss")

    plt.subplot(1, 3, 3)
    plt.plot(agent.expected_values)
    plt.title("Expected Values Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Expected Value")
    plt.savefig('/Users/yuyanlin/Desktop/AdaptiveTrafficLight/changing-epsilon.png')


print(main())
