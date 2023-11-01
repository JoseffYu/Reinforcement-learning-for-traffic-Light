import gym

# build env
env = gym.make("CartPole-v1")

# initialise env
env.reset()
done = False
total_reward = 0
while not done:
   env.render()
   #Learning，do actions randomly，（action）
   step_result = env.step(env.action_space.sample())
   # Here different from tutorial, add one more assignment to get value
   obs, rew, done, info = step_result[:4]
   total_reward += rew
   print(f"{obs} -> {rew}")
   print(f"done: {done}")
print(f"Total reward: {total_reward}")
