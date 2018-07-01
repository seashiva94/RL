import tensorflow as tf
import cv2
import numpy as np
import random
from pong import Pong
from collections import deque

# deep q network for pong game

# define hyper params
ACTIONS = 3
GAMMA = 0.99
INITIAL_EPS = 1.0
FINAL_EPS = 0.05
EXPLORE = 500000
OBSERVE = 50000
REPLAY_MEMORY = 50000
BATCH = 100


#create tf graph
def createGraph():
    # cobnv1, bias
    W_conv1 = tf.Variable(tf.zeros([8,8,4,32]))
    b_conv1 = tf.Variable(tf.zeros([32]))

    # second layer
    W_conv2 = tf.Variable(tf.zeros([4,4,32,64]))
    b_conv2 = tf.Variable(tf.zeros([64]))

    W_conv3 = tf.Variable(tf.zeros([3,3,64,64]))
    b_conv3 = tf.Variable(tf.zeros([64]))

    W_fc4 = tf.Variable(tf.zeros([3136,784]))
    b_fc4 = tf.Variable(tf.zeros([784]))

    W_fc5 = tf.Variable(tf.zeros([784,ACTIONS]))
    b_fc5 = tf.Variable(tf.zeros([ACTIONS]))


    # input
    s = tf.placeholder("float", [None, 84, 84, 4])

    conv1 = tf.nn.relu(tf.nn.conv2d(s, W_conv1, strides=[1,4,4,1], padding = "VALID") + b_conv1)
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W_conv2, strides=[1,2,2,1], padding = "VALID") + b_conv2)
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, W_conv3, strides=[1,1,1,1], padding = "VALID") + b_conv3)

    conv3_flat = tf.reshape(conv3, [-1, 3136])
    fc4 = tf.nn.relu(tf.matmul(conv3_flat, W_fc4) + b_fc4)
    fc5 = tf.matmul(fc4, W_fc5) + b_fc5
    return s, fc5


def trainGraph(inp, out, sess):
    argmax = tf.placeholder("float", [None, ACTIONS])
    gt = tf.placeholder("float", [None]) # ground truth

    action = tf.reduce_sum(tf.multiply(out, argmax), reduction_indices = 1)
    cost = tf.reduce_mean(tf.square(action - gt))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # initialize pong
    game = Pong()

    D = deque()

    frame = game.getPresentFrame()
    frame = cv2.cvtColor(cv2.resize(frame, (84,84)), cv2.COLOR_BGR2GRAY)
    ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    inp_t = np.stack((frame, frame, frame, frame), axis = 2)

    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())

    t = 0
    epsilon = INITIAL_EPS

    # training the network
    while(True):
        out_t = out.eval(feed_dict = {inp: [inp_t]})[0]
        argmax_t = np.zeros([ACTIONS])

        # explore random action with some small prob
        if(random.random() <= epsilon):
            maxIndex = random.randrange(ACTIONS)
        else:
            maxIndex = np.argmax(out_t)
            
        argmax_t[maxIndex] = 1

        # decay epsilon oover time
        if epsilon > FINAL_EPS:
            epsilon -= (INITIAL_EPS - FINAL_EPS)/EXPLORE

        # get reward for the next frame
        reward_t, frame = game.getNextFrame(argmax_t)
        frame = cv2.cvtColor(cv2.resize(frame, (84,84)), cv2.COLOR_BGR2GRAY)
        ret, frame = cv2.threshold(frame, 1,255, cv2.THRESH_BINARY)
        frame = np.reshape(frame, (84,84,1))

        inp_t1 = np.append(frame, inp_t[:,:,0:3], axis = 2)
        # append input, argmax, reward and next input to wueue
        D.append((inp_t, argmax_t, reward_t, inp_t1))

        if len(D) > REPLAY_MEMORY:
            D.popleft()

        if t > OBSERVE:
            minibatch = random.sample(D, BATCH)
            inp_batch = [d[0] for d in minibatch]
            argmax_batch = [d[1] for d in minibatch]
            reward_batch = [d[2] for d in minibatch]
            inp_t1_batch = [d[3] for d in minibatch]

            gt_batch = []
            out_batch = out.eval(feed_dict = {inp: inp_t1_batch})

            for i in range(0, len(minibatch)):
                gt_batch.append(reward_batch[i] + gamma*mp.max(out_batch[i]))

            train_step.run(feed_dict = {
                gt: gt_batch,
                argmax : argmax_batch,
                inp : inp_batch
            })

        inp_t = inp_t1
        t = t +1

        if t%10000 == 0:
            saver.save(sess, './pong-dqn', global_step = t)

        print("TIMESTEP", t, " / EPSILON ", epsilon, " / ACTION", maxIndex, " / REWARD ", reward_t, " /Qmax %e"%np.max(out_t))

            
def main():
    sess = tf.InteractiveSession()
    inp, out = createGraph()
    trainGraph(inp, out, sess)


if __name__ == "__main__":
    main()
