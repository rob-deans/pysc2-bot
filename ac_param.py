import tensorflow as tf
import random
from collections import deque


class ActorCriticModelCont:
    def __init__(self, input_size, input_flat, action_size, actor_lr, critic_lr, memory, gamma):
        self.wh = input_size
        self.input_flat = input_flat
        self.num_actions = action_size
        self.gamma = gamma
        self.memory = memory

        # Generic
        init = tf.truncated_normal_initializer(0., .01)

        # ===================================== #
        #               Actor                   #
        # ===================================== #

        self.input = tf.placeholder(tf.float32, shape=[None, self.input_flat], name='input')
        self.actor_actions = tf.placeholder(tf.float32, shape=[None, self.num_actions], name='actions')
        self.td_error = tf.placeholder(tf.float32, shape=[None, 1], name='rewards')

        self.actor_lr = actor_lr

        image = tf.reshape(self.input, [-1, self.wh, self.wh, 1])

        conv_net = tf.layers.conv2d(inputs=image, filters=16, kernel_size=5, padding='same', activation=tf.nn.relu)
        conv_net = tf.layers.conv2d(inputs=conv_net, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
        #
        conv_net = tf.contrib.layers.flatten(conv_net)

        net = tf.layers.dense(inputs=conv_net, units=256, activation=tf.nn.relu, kernel_initializer=init, name='dense1')
        # net = tf.layers.dense(inputs=net, units=128, activation=tf.nn.relu, kernel_initializer=init, name='dense2')

        self.output = tf.layers.dense(inputs=net, units=self.num_actions, activation=tf.nn.softmax)

        loss = tf.log(tf.reduce_sum(tf.multiply(self.output, self.actor_actions))) * self.td_error
        self.optimiser = tf.train.AdamOptimizer(self.actor_lr).minimize(-loss)

        # ===================================== #
        #                Critic                 #
        # ===================================== #

        # self.critic_input = tf.placeholder(tf.float32, shape=[None, self.input_flat], name='critic_input')
        self.critic_td_target = tf.placeholder(tf.float32, shape=[None, 1], name='rewards')
        self.critic_lr = critic_lr

        critic_net = tf.layers.dense(inputs=conv_net, units=256, activation=tf.nn.relu, kernel_initializer=init)
        self.critic_output = tf.layers.dense(inputs=critic_net, units=1, activation=None, kernel_initializer=init)

        self.critic_loss = tf.reduce_mean(tf.squared_difference(self.critic_output, self.critic_td_target))
        self.critic_optimise = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss)

        self.saver = tf.train.Saver()

        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.session.run(tf.global_variables_initializer())
        try:
            self.saver.restore(self.session, '/home/rob/Documents/uni/fyp/sc2/ac_model_param.ckpt')
        except:
            print('No model found - training new one')

    def train(self):
        if len(self.memory.memory) < 16:
            return

        td_targets = []
        td_errors = []

        mini_batch = self.memory.get_batch(self.memory.batch_size)

        states = [item[0] for item in mini_batch]
        actions = [item[1] for item in mini_batch]
        rewards = [item[2] for item in mini_batch]
        done = [item[3] for item in mini_batch]
        next_states = [item[4] for item in mini_batch]

        values = self.batch_predict(states)
        # next_values = self.batch_predict(next_states)

        for i in range(len(mini_batch)):
            if done[i]:
                td_targets.append([rewards[i]])
            else:
                td_targets.append(rewards[i] + self.gamma * self.predict(next_states[i]))

            td_errors.append(td_targets[-1] - values[i])

        # Training the critic
        self.session.run(self.critic_optimise, feed_dict={
            self.input: states,
            self.critic_td_target: td_targets
        })

        # Training the actor
        self.session.run(self.optimiser, feed_dict={
            self.input: states,
            self.actor_actions: actions,
            self.td_error: td_errors
        })
        del self.memory.memory[:]

    def run(self, state):
        return self.session.run(self.output, feed_dict={self.input: [state]})[0]

    def batch_predict(self, states):
        return self.session.run(self.critic_output, feed_dict={self.input: states})

    def predict(self, state):
        return self.session.run(self.critic_output, feed_dict={self.input: [state]})[0]

    def save(self):
        self.saver.save(self.session, '/home/rob/Documents/uni/fyp/sc2/ac_model_param.ckpt')


class ReplayMemory:
    def __init__(self, batch_size, max_memory_size):
        self.batch_size = batch_size
        self.memory = []

    # Add the current state, the action we took, the reward we got for it and
    # whether it was the terminal (done) state for the ep
    def add(self, state, action, reward, done):
        self.update(state)
        if len(self.memory) == 15:
            done = True
        self.memory.append([state, action, reward, done, None])
        if reward == 1.:
            self.memory.reverse()
            running_add = 0
            for m, mem in enumerate(self.memory):
                _, _, r, d, _ = mem
                if done and m > 0:
                    break
                running_add = running_add * 0.99 + r
                mem[3] = running_add
            self.memory.reverse()

    # Update the memory to include the next state
    def update(self, next_state):
        if len(self.memory) > 0 and not self.memory[-1][3]:
            self.memory[-1][4] = next_state

    def get_batch(self, batch_size=50):
        return self.memory
        # temp = self.memory
        # temp.pop()
        # return random.sample(temp, batch_size - 1)
