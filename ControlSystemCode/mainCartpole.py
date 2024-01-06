
import gym
import matplotlib
from CartpoleMethod1 import DQN, ReplayBuffer
import matplotlib.pyplot as plt
import torch
from itertools import count

device = "cpu"
sumo_cfg = "/Users/yuyanlin/Desktop/AdaptiveTrafficLight/Simulator/RunSimulator.sumocfg"

all_total_reward = []

plt.ion()


def main():
    
    env = gym.make("CartPole-v1",render_mode = 'human')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQN(env = env,mode="train",input_dim=state_size,output_dim=action_size,gamma=0.95,replay_size=20000,batch_size=32,eps_start=0.95,eps_end=0.05,eps_decay=30000,TAU=0.005,LR=1e-4)
    replay_buffer = ReplayBuffer(agent.replay_size)
    agent.memory = replay_buffer
    
    for episode in range(1000):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            env.render()
            action = agent.selectAction(state,agent.learn_step_counter)
            observation, reward, done, _ = env.step(action.item())[:4]
            reward = torch.tensor([reward], device=device)

            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            agent.memory.push(state, action, next_state, reward)
            state = next_state
            
            agent.learn()

            for param in agent.policy_net_state_dict:
                agent.target_net_state_dict[param] = agent.policy_net_state_dict[param]*agent.TAU + agent.policy_net_state_dict[param]*(1-agent.TAU)

            agent.target_net.load_state_dict(agent.target_net_state_dict)
            
            if done:
                agent.episode_durations.append(t+1)
                agent.plot_durations()
                break
        print('i_episode:', episode)
        print('learn_steps:', agent.learn_step_counter)
    
    all_total_reward = agent.episode_durations
    print("Done")
    
    
            
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(torch.tensor(all_total_reward, dtype=torch.float))
    plt.title('Returns for Cartpole')
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
    plt.xlabel("learn_steps")
    plt.ylabel("Expected Value")
    plt.savefig('/Users/yuyanlin/Desktop/AdaptiveTrafficLight/Cartpole_method1_reward_epi.png')

plt.show()

print(main())
