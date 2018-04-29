import tensorflow as tf
import numpy as np
import random


class Model:
    def __init__(self, input_flat, action_size, learning_rate, memory):
        self.input_flat = input_flat
        self.num_actions = action_size
        self.memory = memory

        self.screen_input = tf.placeholder(tf.float32, shape=[None, self.input_flat], name='input')
        self.actions = tf.placeholder(tf.float32, shape=[None, self.num_actions], name='actions')

        init = tf.random_normal_initializer

        # create the network
        net = self.screen_input

        hidden1 = tf.contrib.layers.fully_connected(
            inputs=net,
            num_outputs=512,
            activation_fn=tf.nn.relu,
            weights_initializer=init
        )

        hidden2 = tf.contrib.layers.fully_connected(
            inputs=hidden1,
            num_outputs=256,
            activation_fn=tf.nn.relu,
            weights_initializer=init
        )

        logits = tf.contrib.layers.fully_connected(
            inputs=hidden2,
            num_outputs=self.num_actions,
            activation_fn=tf.nn.softmax
        )

        self.output = logits

        # training part of graph
        self._acts = tf.placeholder(tf.float32)
        self._advantages = tf.placeholder(tf.float32)

        # loss function
        loss = tf.log(tf.reduce_sum(tf.multiply(self._acts, self.output))) * self._advantages
        self._train = tf.train.AdamOptimizer(learning_rate).minimize(-loss)

        self.saver = tf.train.Saver()
        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.session.run(tf.global_variables_initializer())

    def train(self):
        print('== TRAINING ==')
        states, actions, advantages = self.memory.get()
        self.session.run(self._train, feed_dict={
                            self.screen_input: states,
                            self._acts: actions,
                            self._advantages: advantages
        })
        self.memory.delete()

    def get_action(self, state):
        return self.session.run(self.output, feed_dict={self.screen_input: [state]})

    def save(self):
        self.saver.save(self.session, './policy_model.ckpt')


class ReplayMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.advantages = []

    # Add the run to the memory
    def add(self, states, actions, rewards):
        self.states.extend(states)
        self.actions.extend(actions)
        self.advantages.extend(rewards)

    # Get the full run
    def get(self):
        # Normalise the rewards
        self.advantages = (self.advantages - np.mean(self.advantages)) // (np.std(self.advantages) + 1e-10)
        return self.states, self.actions, self.advantages

    # Delete the runs after each training session
    def delete(self):
        del self.states[:]
        del self.actions[:]
        self.advantages = []

