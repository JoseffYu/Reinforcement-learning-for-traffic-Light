
import numpy as np
from statistics import mean
from SUMOenv import SumoEnv
from ActorCritic import Actor,Critic
from PER import DQN, ReplayBuffer_Per
#from DDqnChangeEpsilon import DQN, ReplayBuffer
import matplotlib.pyplot as plt   
import torch


device = "cpu"
sumo_cfg = "/Users/yuyanlin/Desktop/AdaptiveTrafficLight/Simulator/SimpleEnv.sumocfg"


all_total_reward = []
num_vehicles_pass = []
average_vehicle_passed = []
        
actor_losses = []
cric_losses = []


def main():

    env = SumoEnv(sumo_cfg_file=sumo_cfg,
            simulation_time=500
            )

    state_size = env.observation_space().shape[0]
    state_ex = env.observation_space()
    action_size = env.action_space().n
    lr = 0.0001
    
    actor = Actor(state_size,action_size).to(device)
    critic = Critic(state_size,action_size).to(device)
    
    optimizerA = torch.optim.Adam(actor.parameters())
    optimizerC = torch.optim.Adam(critic.parameters())
    for iter in range(100):
        log_probs = []
        
        values = []
        rewards = []
        masks = []
        entropy = 0
        total_reward = 0
        total_arrived_vehicle = 0
        env.reset()
        state = env.computeState()
        done = False
        
        while not done:
            state = np.nan_to_num(state, nan=0)
            state = torch.from_numpy(state)
            dist, value = actor.forward(state), critic.forward(state)
            action = dist.sample()
            state, next_state, reward, done, info = env.step(env.tl_id,action.cpu().numpy())

            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))
            total_reward += reward
            total_arrived_vehicle += env.traffic_light.conn.simulation.getArrivedNumber()

            state = next_state
            
        env.close()
        
        
        next_state = torch.FloatTensor(next_state).to(device)
        next_value = critic(next_state)
        returns = critic.discounted_rewards(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        
        actor_losses.append(actor_loss.item())
        cric_losses.append(critic_loss.item())

        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()
        
        all_total_reward.append(total_reward)
        num_vehicles_pass.append(total_arrived_vehicle)
        average_vehicle_passed.append(mean(num_vehicles_pass))
    
        print(actor_losses)
        print('episode_return: ',total_reward)
        print('arrived_vehicles: ',total_arrived_vehicle)
        print('i_episode:', iter)

    
    print(mean(num_vehicles_pass))
    
    print(all_total_reward)
    p1 = plt
    p1.subplots()
    p1.plot(all_total_reward)
    p1.title('Returns for DDQN')
    p1.xlabel('Episodes')
    p1.ylabel('returns')
    p1.grid()
    p1.savefig('/Users/yuyanlin/Desktop/AdaptiveTrafficLight/Experiments/ActorCritic/Returns_for_AC2.png')

    p2 = plt
    p2.subplots()
    print(num_vehicles_pass)
    x = np.arange(len(num_vehicles_pass))
    y = np.ones_like(x) * 400
    p2.plot(x,y)
    p2.plot(num_vehicles_pass)
    p2.plot(average_vehicle_passed)
    p2.title("Vehicles passed over time")
    p2.xlabel("Episode")
    p2.ylabel("number of vehicles arrived destination")
    p2.legend(['base line','number of vehicle pass','average vehicles pass'])
    p2.grid()
    p2.savefig('/Users/yuyanlin/Desktop/AdaptiveTrafficLight/Experiments/ActorCritic/arrived_vehicles22.png')
    
    p3 = plt
    p3.subplots()
    p3.plot(actor_losses)
    p3.title("Actor losses Over Time")
    p3.xlabel("episodes")
    p3.ylabel("Loss")
    p3.grid()
    p3.savefig('/Users/yuyanlin/Desktop/AdaptiveTrafficLight/Experiments/ActorCritic/aloss_for_AC2.png')
    
    p4 = plt
    p4.subplots()
    p4.plot(cric_losses)
    p4.title("Critic losses Over Time")
    p4.xlabel("episodes")
    p4.ylabel("Loss")
    p4.grid()
    p4.savefig('/Users/yuyanlin/Desktop/AdaptiveTrafficLight/Experiments/ActorCritic/closs_for_AC2.png')

print(main())
