import tensorflow as tf
import numpy as np
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
        self.minimap_input = tf.placeholder(tf.float32, shape=[None, self.minimap_flat], name='minimap_input')
        self.army_input = tf.placeholder(tf.float32, shape=[None, 1], name='army_input')
        self.actions = tf.placeholder(tf.float32, shape=[None, self.num_actions], name='actions')

        x_image = tf.reshape(self.screen_input, [-1, self.wh, self.wh, 1])
        mm_image = tf.reshape(self.minimap_input, [-1, self.mm_wh, self.mm_wh, 1])

        init = tf.random_normal_initializer

        # create the network
        net = x_image

        net = tf.layers.conv2d(inputs=net, filters=8, kernel_size=5, padding='same', activation=tf.nn.relu)
        net = tf.layers.conv2d(inputs=net, filters=16, kernel_size=5, padding='same', activation=tf.nn.relu)

        net = tf.contrib.layers.flatten(net)

        mm_image = tf.layers.conv2d(inputs=mm_image, filters=4, kernel_size=5, padding='same', activation=tf.nn.relu)
        mm_image = tf.layers.conv2d(inputs=mm_image, filters=8, kernel_size=5, padding='same', activation=tf.nn.relu)

        mm_image = tf.contrib.layers.flatten(mm_image)

        x_army = tf.layers.dense(inputs=self.army_input, units=9, activation=tf.nn.relu, kernel_initializer=init)

        dense_1 = tf.concat([net, mm_image, x_army], 1)

        hidden1 = tf.contrib.layers.fully_connected(
            inputs=dense_1,
            num_outputs=36,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.random_normal_initializer
        )

        logits = tf.contrib.layers.fully_connected(
            inputs=hidden1,
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
        states, minimaps, army_selected, actions, advantages = self.memory.get()
        self.session.run(self._train, feed_dict={
                            self.screen_input: states,
                            self.minimap_input: minimaps,
                            self.army_input: army_selected,
                            self._acts: actions,
                            self._advantages: advantages
        })
        self.memory.delete()

    def get_action(self, state, minimap, army):
        return self.session.run(self.output, feed_dict={self.screen_input: [state],
                                                        self.minimap_input: [minimap],
                                                        self.army_input: [army]}
                                )

    def save(self):
        self.saver.save(self.session, './policy_model.ckpt')


class ReplayMemory:
    def __init__(self):
        self.states = []
        self.minimap_states = []
        self.army_selected = []
        self.actions = []
        self.advantages = []

    # Add the run to the memory
    def add(self, states, minimap_states, army_state, actions, rewards):
        self.states.extend(states)
        self.minimap_states.extend(minimap_states)
        self.army_selected.extend(army_state)
        self.actions.extend(actions)
        self.advantages.extend(rewards)

    # Get the full run
    def get(self):
        # Normalise the rewards
        self.advantages = (self.advantages - np.mean(self.advantages)) // (np.std(self.advantages) + 1e-10)
        return self.states, self.minimap_states, self.army_selected, self.actions, self.advantages

    # Delete the runs after each training session
    def delete(self):
        del self.states[:]
        del self.minimap_states[:]
        del self.army_selected[:]
        del self.actions[:]
        self.advantages = []

