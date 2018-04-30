import tensorflow as tf
import numpy as np
from collections import deque
import random
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.agents import base_agent

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_MM_PLAYER_RELATIVE = features.MINIMAP_FEATURES.player_relative.index
_SELECT = features.SCREEN_FEATURES.selected.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4

_NOT_QUEUED = [0]
_SELECT_ALL = [0]

_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_CAMERA = actions.FUNCTIONS.move_camera.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_RECT = actions.FUNCTIONS.select_rect.id
_SELECT_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id
_STOP_QUICK = actions.FUNCTIONS.Stop_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_MOVE_MINIMAP = actions.FUNCTIONS.Move_minimap.id
_PATROL_SCREEN = actions.FUNCTIONS.Patrol_screen.id
_PATROL_MINIMAP = actions.FUNCTIONS.Patrol_minimap.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_HOLD_POSITION_QUICK = actions.FUNCTIONS.HoldPosition_quick.id
_SMART_SCREEN = actions.FUNCTIONS.Smart_screen.id
_SMART_MINIMAP = actions.FUNCTIONS.Smart_minimap.id

# All the available actions
available_actions = [
    _NO_OP,
    _MOVE_CAMERA,
    _SELECT_POINT,
    _SELECT_RECT,
    _SELECT_CONTROL_GROUP,
    _STOP_QUICK,
    _SELECT_ARMY,
    _ATTACK_SCREEN,
    _MOVE_SCREEN,
    _MOVE_MINIMAP,
    _PATROL_SCREEN,
    _PATROL_MINIMAP,
    _ATTACK_MINIMAP,
    _HOLD_POSITION_QUICK,
    _SMART_SCREEN,
    _SMART_MINIMAP
]


class ActorCriticModel:
    def __init__(self, input_size, input_flat, minimap_size, minimap_flat,
                 action_size, actor_lr, critic_lr, memory, gamma):
        self.wh = input_size
        self.input_flat = input_flat
        self.mm_wh = minimap_size
        self.minimap_flat = minimap_flat
        self.num_actions = action_size
        self.gamma = gamma
        self.memory = memory

        # Generic
        init = tf.truncated_normal_initializer(0, 0.01)

        # ===================================== #
        #               Actor                   #
        # ===================================== #

        self.input = tf.placeholder(tf.float32, shape=[None, self.input_flat], name='screen_input')
        self.minimap_input = tf.placeholder(tf.float32, shape=[None, self.minimap_flat], name='mini_input')
        self.army_selected = tf.placeholder(tf.float32, shape=[None, self.input_flat], name='army_input')
        self.actor_actions = tf.placeholder(tf.float32, shape=[None, self.num_actions], name='actions')
        self.td_error = tf.placeholder(tf.float32, shape=[None, 1], name='rewards')

        self.actor_lr = actor_lr

        image = tf.reshape(self.army_selected, [-1, self.wh, self.wh, 3])

        net = tf.contrib.layers.conv2d(inputs=image, num_outputs=16, kernel_size=5, padding='same',
                                       activation_fn=tf.nn.relu)
        conv_net = tf.contrib.layers.conv2d(inputs=net, num_outputs=32, kernel_size=3, padding='same',
                                            activation_fn=tf.nn.relu)

        logits = tf.contrib.layers.conv2d(conv_net, num_outputs=1, kernel_size=1, activation_fn=None)

        self.output = tf.nn.softmax(tf.contrib.layers.flatten(logits))

        loss = tf.log(tf.reduce_sum(tf.multiply(self.output, self.actor_actions))) * self.td_error
        # entropy = tf.reduce_sum(tf.multiply(self.output, tf.log(self.output)))
        # loss += 0.001 * entropy

        self.optimiser = tf.train.AdamOptimizer(self.actor_lr).minimize(-loss)

        # ===================================== #
        #                Critic                 #
        # ===================================== #

        self.critic_td_target = tf.placeholder(tf.float32, shape=[None, 1], name='rewards')
        self.critic_lr = critic_lr

        critic_net = tf.layers.dense(inputs=conv_net, units=256, activation=tf.nn.relu, kernel_initializer=init)
        self.critic_output = tf.layers.dense(inputs=critic_net, units=1, activation=None, kernel_initializer=init)

        self.critic_loss = tf.reduce_mean(tf.squared_difference(self.critic_output, self.critic_td_target))
        self.critic_optimise = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss)

        self.saver = tf.train.Saver()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        try:
            self.saver.restore(self.session, '/home/rob/Documents/uni/fyp/sc2/ac_model.ckpt')
        except:
            print('No model found - training new one')

    def train(self):
        if len(self.memory.memory) < self.memory.batch_size:
            return

        mini_batch = self.memory.get_batch()

        td_targets = []
        td_errors = []

        army = [item[0] for item in mini_batch]
        actions = [item[1] for item in mini_batch]
        rewards = [item[2] for item in mini_batch]
        done = [item[3] for item in mini_batch]
        next_states = [item[4] for item in mini_batch]

        values = self.batch_predict(army)

        for i in range(len(done)):
            if done[i]:
                td_targets.append([rewards[i]])
            else:
                td_targets.append(rewards[i] + self.gamma * self.predict(next_states[i]))

            td_errors.append(td_targets[-1] - values[i])

        # Training the critic
        self.session.run(self.critic_optimise, feed_dict={
            self.army_selected: army,
            self.critic_td_target: td_targets
        })

        # Training the actor
        self.session.run(self.optimiser, feed_dict={
            self.army_selected: army,
            self.actor_actions: actions,
            self.td_error: td_errors
        })

    def run(self, army):
        return self.session.run(self.output, feed_dict={self.army_selected: [army]})[0]

    def batch_predict(self, army):
        return self.session.run(self.critic_output, feed_dict={self.army_selected: army})

    def predict(self, next_army):
        return self.session.run(self.critic_output, feed_dict={self.army_selected: [next_army]})[0]

    def save(self):
        self.saver.save(self.session, '/home/rob/Documents/uni/fyp/sc2/ac_model.ckpt')


class ReplayMemory:
    def __init__(self, batch_size, max_memory_size=2000):
        self.batch_size = batch_size
        self.memory = deque(maxlen=max_memory_size)

    def add(self, state, action, reward, done):
        self.memory.append([state, action, reward, done, None])

    def update(self, next_state):
        if len(self.memory) > 0:
            if not self.memory[-1][3]:
                self.memory[-1][4] = next_state

    def get_batch(self):
        return random.sample(self.memory, self.batch_size)

