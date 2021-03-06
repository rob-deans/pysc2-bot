import tensorflow as tf
import numpy as np
from collections import deque
import random


class Model:
    def __init__(self, input_size, input_flat, minimap_size, minimap_flat,
                 army_input, action_size, learning_rate, memory):
        self.wh = input_size
        self.input_flat = input_flat
        self.mm_wh = minimap_size
        self.minimap_flat = minimap_flat
        self.army_input = army_input
        self.num_actions = action_size
        self.memory = memory

        self.screen_input = tf.placeholder(tf.float32, shape=[None, self.input_flat], name='input')
        self.minimap_input = tf.placeholder(tf.float32, shape=[None, self.minimap_flat], name='mm_input')
        self.army_input = tf.placeholder(tf.float32, shape=[None, self.input_flat], name='army_input')

        self.actions = tf.placeholder(tf.float32, shape=[None, self.num_actions], name='actions')
        self.rewards = tf.placeholder(tf.float32, shape=[None], name='rewards')

        x_image = tf.reshape(self.army_input, [-1, self.wh, self.wh, 1])
        # mm_image = tf.reshape(self.minimap_input, [-1, self.mm_wh, self.mm_wh, 1])

        init = tf.truncated_normal_initializer(0, 0.01)

        # create the network
        net = x_image

        net = tf.layers.conv2d(inputs=net, filters=8, kernel_size=5, padding='same', activation=tf.nn.relu)
        net = tf.layers.conv2d(inputs=net, filters=16, kernel_size=5, padding='same', activation=tf.nn.relu)

        net = tf.contrib.layers.flatten(net)

        net = tf.layers.dense(inputs=net, units=64, activation=tf.nn.relu, kernel_initializer=init, name='dense1')

        net = tf.layers.dense(inputs=net, units=self.num_actions, activation=None, kernel_initializer=init)

        self.output = net

        q_reward = tf.reduce_sum(tf.multiply(self.output, self.actions), 1)
        loss = tf.reduce_mean(tf.squared_difference(self.rewards, q_reward))
        self.optimiser = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        self.saver = tf.train.Saver()

        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.session.run(tf.global_variables_initializer())
        try:
            self.saver.restore(self.session, '/home/rob/Documents/uni/fyp/sc2/model.ckpt')
            self.loaded_model = True
        except:
            print('No model found - training new one')
            self.loaded_model = False

    def train(self):
        if len(self.memory.memory) - 1 < self.memory.batch_size:
            return
        states, minimaps, army_selected, actions, rewards = self.memory.get_batch(self, self.memory.batch_size)
        self.session.run(self.optimiser, feed_dict={
            self.army_input: army_selected,
            self.actions: actions,
            self.rewards: rewards}
                         )

    def get_action(self, state):
        screen, mini_map, army = state
        feed_dict = {self.army_input: [army]}
        return self.session.run(self.output, feed_dict)[0]

    def get_batch_action(self, states, minimap, army_selected):
        return self.session.run(self.output, feed_dict={self.screen_input: states, self.minimap_input: minimap,
                                                        self.army_input: army_selected})

    def save(self):
        self.saver.save(self.session, '/home/rob/Documents/uni/fyp/sc2/model.ckpt')


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
        self.update(state)
        self.memory.append([state, actions, reward, done, None])

    # Update the memory to include the next state
    def update(self, next_state):
        if len(self.memory) > 0:
            if not self.memory[-1][3]:
                self.memory[-1][4] = next_state

    def get_batch(self, model, batch_size=50):
        mini_batch = np.array(random.sample(self.memory, batch_size))

        states = [item[0][0] for item in mini_batch]
        minimap_states = [item[0][1] for item in mini_batch]
        army_selected = [item[0][2] for item in mini_batch]

        actions = [item[1] for item in mini_batch]
        rewards = [item[2] for item in mini_batch]
        done = [item[3] for item in mini_batch]

        next_states = mini_batch[:, 4]

        y_batch = []

        for i in range(batch_size):
            if done[i]:
                y_batch.append(rewards[i])
            else:
                y_batch.append(rewards[i] + self.gamma * np.max(model.get_action(next_states[i])))

        return states, minimap_states, army_selected, actions, y_batch

