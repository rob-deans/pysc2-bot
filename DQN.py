import tensorflow as tf
import numpy as np
from collections import deque
import random


class Model:
    def __init__(self, input_size, input_flat, action_size, learning_rate, memory):
        self.wh = input_size
        self.input_flat = input_flat
        self.num_actions = action_size
        self.memory = memory

        self.screen_input = tf.placeholder(tf.float32, shape=[None, self.input_flat], name='input')
        self.actions = tf.placeholder(tf.float32, shape=[None, self.num_actions], name='actions')
        self.rewards = tf.placeholder(tf.float32, shape=[None], name='rewards')

        x_image = tf.reshape(self.screen_input, [-1, self.wh, self.wh, 1])

        init = tf.truncated_normal_initializer()

        # create the network
        net = x_image

        net = tf.layers.conv2d(inputs=net, filters=16, kernel_size=5, padding='same', activation=tf.nn.relu)
        # net = tf.layers.conv2d(inputs=net, filters=36, kernel_size=5, padding='same', activation=tf.nn.relu)
        # net = tf.layers.conv2d(inputs=net, filters=36, kernel_size=16, padding='same', activation=tf.nn.relu)

        net = tf.contrib.layers.flatten(net)

        net = tf.layers.dense(inputs=net, units=64, activation=tf.nn.relu, kernel_initializer=init, name='dense1')

        net = tf.layers.dense(inputs=net, units=self.num_actions, activation=None, kernel_initializer=init)

        self.output = net

        q_reward = tf.reduce_sum(tf.multiply(self.output, self.actions), 1)
        loss = tf.reduce_mean(tf.squared_difference(self.rewards, q_reward))
        self.optimiser = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.session.run(tf.global_variables_initializer())

    def train(self):
        if len(self.memory.memory) < self.memory.batch_size:
            return
        states, actions, rewards = self.memory.get_batch(self, self.memory.batch_size)
        self.session.run(self.optimiser, feed_dict={self.screen_input: states, self.actions: actions, self.rewards: rewards})

    def get_action(self, state, epsilon):
        if random.random() < epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            temp = self.session.run(self.output, feed_dict={self.screen_input: [state]})[0]
            action = np.argmax(temp)

        return action

    def get_batch_action(self, states):
        return self.session.run(self.output, feed_dict={self.screen_input: states})


class ReplayMemory:
    def __init__(self, num_actions, batch_size, max_memory_size, gamma):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.memory = deque(maxlen=max_memory_size)
        self.gamma = gamma

    # Add the current state, the action we took, the reward we got for it and
    # whether it was the terminal (done) state for the ep
    def add(self, state, action, reward, done):
        actions = np.zeros(self.num_actions)
        actions[action] = 1
        self.memory.append([state, actions, reward, done, None])
        self.update(state)

    # Update the memory to include the next state
    def update(self, next_state):
        if len(self.memory) > 0:
            self.memory[-1][4] = next_state

    def get_batch(self, model, batch_size=50):
        mini_batch = random.sample(self.memory, batch_size)
        states = [item[0] for item in mini_batch]
        actions = [item[1] for item in mini_batch]
        rewards = [item[2] for item in mini_batch]
        done = [item[3] for item in mini_batch]
        next_states = [item[4] for item in mini_batch]

        q_values = model.get_batch_action(next_states)
        y_batch = []

        for i in range(batch_size):
            if done[i]:
                y_batch.append(rewards[i])
            else:
                y_batch.append(rewards[i] + self.gamma * np.max(q_values[i]))

        return states, actions, y_batch

