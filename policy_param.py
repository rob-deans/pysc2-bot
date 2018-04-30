import tensorflow as tf
import numpy as np
import random


class Model:
    def __init__(self, wh, input_flat, action_size, learning_rate, memory):
        self.wh = wh
        self.input_flat = input_flat
        self.num_actions = action_size
        self.memory = memory

        self.screen_input = tf.placeholder(tf.float32, shape=[None, self.input_flat], name='input')
        self.actions = tf.placeholder(tf.float32, shape=[None, self.num_actions], name='actions')

        init = tf.random_normal_initializer

        image = tf.reshape(self.screen_input, [-1, self.wh, self.wh, 1])

        net = tf.contrib.layers.conv2d(inputs=image, num_outputs=16, kernel_size=5, padding='same', activation_fn=tf.nn.relu)
        conv_net = tf.contrib.layers.conv2d(inputs=net, num_outputs=32, kernel_size=3, padding='same', activation_fn=tf.nn.relu)

        logits = tf.contrib.layers.conv2d(conv_net, num_outputs=1, kernel_size=1, activation_fn=None)

        self.output = tf.nn.softmax(tf.contrib.layers.flatten(logits))

        # training part of graph
        self._acts = tf.placeholder(tf.float32)
        self._advantages = tf.placeholder(tf.float32)

        # loss function
        loss = tf.log(tf.reduce_sum(tf.multiply(self._acts, self.output))) * self._advantages
        # entropy = tf.reduce_sum(tf.multiply(self.output, tf.log(tf.clip_by_value(self.output, 1e-12, 1.))))
        # loss += 0.001 * tf.reduce_mean(entropy)
        self._train = tf.train.AdamOptimizer(learning_rate).minimize(-loss)

        self.saver = tf.train.Saver()
        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.session.run(tf.global_variables_initializer())

    def train(self):
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

