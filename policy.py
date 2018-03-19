import tensorflow as tf
import numpy as np
import random


class Model:
    def __init__(self, input_size, input_flat, army_input, action_size, learning_rate, memory):
        self.wh = input_size
        self.input_flat = input_flat
        self.army_input = army_input
        self.num_actions = action_size
        self.memory = memory

        self.screen_input = tf.placeholder(tf.float32, shape=[None, self.input_flat], name='input')
        self.army_input = tf.placeholder(tf.float32, shape=[None, 1], name='army_input')
        self.actions = tf.placeholder(tf.float32, shape=[None, self.num_actions], name='actions')

        x_image = tf.reshape(self.screen_input, [-1, self.wh, self.wh, 1])

        init = tf.random_normal_initializer

        # create the network
        net = x_image

        net = tf.layers.conv2d(inputs=net, filters=16, kernel_size=5, padding='same', activation=tf.nn.relu)
        net = tf.layers.conv2d(inputs=net, filters=32, kernel_size=5, padding='same', activation=tf.nn.relu)

        net = tf.contrib.layers.flatten(net)

        x_army = tf.layers.dense(inputs=self.army_input, units=9, activation=tf.nn.relu, kernel_initializer=init)

        dense_1 = tf.concat([net, x_army], 1)

        hidden1 = tf.contrib.layers.fully_connected(
            inputs=dense_1,
            num_outputs=36,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.random_normal_initializer
        )
        # net = tf.layers.dense(inputs=dense_1, units=32, activation=tf.nn.relu, kernel_initializer=init, name='dense1')

        logits = tf.contrib.layers.fully_connected(
            inputs=hidden1,
            num_outputs=3,
            activation_fn=tf.nn.softmax
        )
        # logits = tf.layers.dense(inputs=net, units=self.num_actions, activation=tf.nn.softmax, kernel_initializer=init)

        self.output = logits

        # op to sample an action
        self._sample = tf.reshape(tf.multinomial(logits, 1), [])
        # potentially do nothing with the logits so that we can do the redistribution later on

        # get log probabilities
        # log_prob = tf.log(logits)

        # training part of graph
        self._acts = tf.placeholder(tf.float32)
        self._advantages = tf.placeholder(tf.float32)

        # get log probabilities of actions from episode
        # indices = tf.range(0, tf.shape(log_prob)[0]) * tf.shape(log_prob)[1] + self._acts
        # act_prob = tf.gather(tf.reshape(log_prob, [-1]), indices)

        loss = tf.log(tf.reduce_sum(tf.multiply(self._acts, self.output))) * self._advantages

        # surrogate loss
        # loss = -tf.reduce_sum(tf.multiply(act_prob, self._advantages))
        self._train = tf.train.AdamOptimizer(learning_rate).minimize(-loss)

        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.session.run(tf.global_variables_initializer())

    def train(self):
        print('== TRAINING ==')
        states, army_selected, actions, advantages = self.memory.get()
        self.session.run(self._train, feed_dict={
                            self.screen_input: states,
                            self.army_input: army_selected,
                            self._acts: actions,
                            self._advantages: advantages
        })
        self.memory.delete()

    def get_action(self, state, army):
        return self.session.run(self.output, feed_dict={self.screen_input: [state], self.army_input: [army]})


class ReplayMemory:
    def __init__(self):
        self.states = []
        self.army_selected = []
        self.actions = []
        self.advantages = []

    # Add the run to the memory
    def add(self, states, army_state, actions, rewards):
        self.states.extend(states)
        self.army_selected.extend(army_state)
        self.actions.extend(actions)
        self.advantages.extend(rewards)

    # Get the full run
    def get(self):
        # Normalise the rewards
        self.advantages = (self.advantages - np.mean(self.advantages)) // (np.std(self.advantages) + 1e-10)
        return self.states, self.army_selected, self.actions, self.advantages

    # Delete the runs after each training session
    def delete(self):
        del self.states[:]
        del self.army_selected[:]
        del self.actions[:]
        self.advantages = []

