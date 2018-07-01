import gym
import numpy as np

# env = gym.make("CartPole-v0")

# hill climbing policy
# init weights randomly
# if weights are good save to memory

def run_episode(env, params):
    observation = env.reset()
    totalreward = 0

    #for 200 timesteps
    for _ in range(200):
      env.render()
      #initialize random weights
      action = 0 if np.matmul(params, observation)<0 else 1
      obeservation, reward, done, info = env.step(action)
      totalreward += reward
      if done:
          break
    return totalreward

def train(submit):
    env = gym.make("CartPole-v0")

    episodes_per_update = 5
    noise_scaling = 0.1

    params = np.random.rand(4)*2 -1
    bestreward = 0

    # 2000 episodes

    for _ in range(2000):
        newparams = params + (np.random.rand(4)*2 -1) * noise_scaling
        reward = run_episode(env, newparams)
        print("reward %d best %d" % (reward, bestreward))
        if reward > bestreward:
            bestreward = reward
            params = newparams
            if reward == 200:
                break


r = train(False)
print(r)
      
