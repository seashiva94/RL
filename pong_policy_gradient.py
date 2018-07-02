import gym
import numpy as np
import pickle

# use policy gradient for pong
# using 2 layer feed forward network

# hyper params
h = 200
batch_size = 10
learning_rate = 1e-4
gamma = 0.99 # discount factor
decay_rate = 0.99
resume = False

# model
D = 80 * 80 # input dimension
if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {}
    # xavier initialization
    model['W1'] = np.random.randn(h,D) / np.sqrt(D)
    model['W2'] = np.random.randn(h) / np.sqrt(h)

grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}
rmsprop_cache = {k: np.zeros_like(v) for k,v in model.items()}

#activation
def sigmoid(x):
    return 1.0 / (1.0 *np.exp(-x))

def preprocess(I):
    # I = game image , crop, downsample
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0 # erase background
    I[I == 109] = 0
    I[I != 0] = 1 # set paddles and balls to 1
    return I.astype(np.float).ravel() # flatten

def discount_reward(r):
    # r = set of rewards
    # more recent rewards are weighted higher
    
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] !=0:
            running_add = 0
        running_add = running_add *gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(x):
     hidden = np.dot(model["W1"],x)
     hidden[hidden<0] = 0 # ReLU shorthand
     logp = np.dot(model["W2"], hidden)
     p = sigmoid(logp)
     return p, hidden

def policy_backward(eph, epdlogp):
    # eph = array of intermediate hidden state = episode
    dw2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model["W2"])
    dh[eph <=0] = 0
    dw1 = np.dot(dh.T, epx)

    return {"W1":dw1, "W2":dw2}

#implement policy gradient
env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

#training
while True:
    curr_x = preprocess(observation)
    x = curr_x - prev_x if prev_x is not None else np.zeros(D) 
    pre_x = curr_x

    #forward prop
    aprob, hidden = policy_forward(x)
    # sample action based on a uniform distribution
    action = 2 if np.random.uniform() < aprob else 3

    xs.append(x)
    hs.append(hidden)
    y = 1 if action == 2 else 0
    dlogps.append(y - aprob)

    # next step
    env.render()
    observation, reward, done, info = env.step(action)
    reward_sum += reward
    
    drs.append(reward)
    if done:
        episode_number += 1

        # stack variables in arrays
        # batch training

        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)

        xs, hs, dlogps, drs = [], [], [] ,[]

        # discount reward computation
        discounted_epr = discount_reward(epr)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp +=  discounted_epr
        grad = policy_backward(eph, epdlogp)
        for k in model:
            grad_buffer[k] += grad[k]

        if episode_number % batch_size == 0:
            for k, v in model.items():
                g = grad_buffer[k]
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1- decay_rate)* g**2
                model[k] +=  learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)

        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print("reset environment, episode total reward was %f, running mean is %f" % (reward_sum, running_reward))
        if episode_number %100 == 0:
            pickle.dump(model, open("save.p", "wb"))

        reward_sum = 0
        observation = env.reset()
        prev_x = None

    if reward !=0:
        print("episode %d finishedwirh reward %d" % (episode_number, reward) , ("" if reward == -1 else "!!!!"))
        
