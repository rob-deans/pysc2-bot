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


class ActorCriticModel():
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
        init = tf.uniform_unit_scaling_initializer

        # ===================================== #
        #               Actor                   #
        # ===================================== #

        self.input = tf.placeholder(tf.float32, shape=[None, self.wh, self.wh, 3], name='screen_input')
        self.minimap_input = tf.placeholder(tf.float32, shape=[None, self.mm_wh, self.mm_wh, 3], name='mini_input')
        self.army_selected = tf.placeholder(tf.float32, shape=[None, self.wh, self.wh, 2], name='army_input')
        self.actor_actions = tf.placeholder(tf.float32, shape=[None, self.num_actions], name='actions')
        self.td_error = tf.placeholder(tf.float32, shape=[None, 1], name='rewards')

        self.actor_lr = actor_lr

        # image = tf.reshape(self.input, [-1, self.wh, self.wh, 1])
        # mm_image = tf.reshape(self.minimap_input, [-1, self.mm_wh, self.mm_wh, 1])
        # a_image = tf.reshape(self.army_selected, [-1, self.wh, self.wh, 1])

        conv_net = tf.layers.conv2d(inputs=self.input, filters=8, kernel_size=5, padding='same', activation=tf.nn.relu)
        conv_net = tf.layers.conv2d(inputs=conv_net, filters=16, kernel_size=3, padding='same', activation=tf.nn.relu)
        #
        conv_net = tf.contrib.layers.flatten(conv_net)

        mm_image = tf.layers.conv2d(inputs=self.minimap_input, filters=4, kernel_size=5, padding='same', activation=tf.nn.relu)
        mm_image = tf.layers.conv2d(inputs=mm_image, filters=8, kernel_size=3, padding='same', activation=tf.nn.relu)

        mm_image = tf.contrib.layers.flatten(mm_image)

        a_image = tf.layers.conv2d(inputs=self.army_selected, filters=4, kernel_size=5, padding='same', activation=tf.nn.relu)
        a_image = tf.layers.conv2d(inputs=a_image, filters=8, kernel_size=3, padding='same', activation=tf.nn.relu)

        a_image = tf.contrib.layers.flatten(a_image)

        combined_net = tf.concat([conv_net, mm_image, a_image], axis=1)

        net = tf.layers.dense(inputs=combined_net, units=256, activation=tf.nn.relu, kernel_initializer=init, name='dense1')

        self.output = tf.layers.dense(inputs=net, units=self.num_actions, activation=tf.nn.softmax)

        loss = tf.reduce_mean(tf.log(tf.reduce_sum(tf.multiply(self.output, self.actor_actions))) * self.td_error)
        # ent = tf.reduce_sum(tf.log(self.output))
        # loss += 0.001 * ent

        self.optimiser = tf.train.AdamOptimizer(self.actor_lr).minimize(-loss)

        # ===================================== #
        #                Critic                 #
        # ===================================== #

        self.critic_td_target = tf.placeholder(tf.float32, shape=[None, 1], name='rewards')
        self.critic_lr = critic_lr

        critic_net = tf.layers.dense(inputs=combined_net, units=256, activation=tf.nn.relu, kernel_initializer=init)
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

        states, mini_map, army, actions, rewards, done = self.memory.get_batch()

        td_targets = []
        td_errors = []

        # states = [item[0][0] for item in mini_batch]
        # mini_map = [item[0][1] for item in mini_batch]
        # army = [item[0][2] for item in mini_batch]
        # actions = [item[1] for item in mini_batch]
        # rewards = [item[2] for item in mini_batch]
        # done = [item[3] for item in mini_batch]

        values = self.batch_predict(states, mini_map, army)

        last_value = self.predict(states[-1], mini_map[-1], army[-1])
        value_target = np.zeros(len(done), dtype=np.float32)
        value_target[-1] = last_value

        # for i in range(len(done)):
        #     value_target[i] = rewards[i] + self.gamma * value_target[i-1]

        for i in range(len(done)):
            if done[i]:
                td_targets.append([rewards[i]])
            else:
                td_targets.append(rewards[i] + self.gamma * values[i+1])

            td_errors.append(td_targets[-1] - values[i])

        # Training the critic
        self.session.run(self.critic_optimise, feed_dict={
            self.input: states,
            self.minimap_input: mini_map,
            self.army_selected: army,
            self.critic_td_target: td_targets
        })

        # Training the actor
        self.session.run(self.optimiser, feed_dict={
            self.input: states,
            self.minimap_input: mini_map,
            self.army_selected: army,
            self.actor_actions: actions,
            self.td_error: td_errors
        })

        # del self.memory.memory[:]
        self.memory.delete()

    def run(self, state, mini_map, army):
        return self.session.run(self.output, feed_dict={self.input: [state], self.minimap_input: [mini_map],
                                                        self.army_selected: [army]})[0]

    def get_action(self, obs):
        obs = obs[0]

        # Current observable state
        screen_player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
        current_state = screen_player_relative.flatten()
        mm_player_relative = obs.observation['minimap'][_MM_PLAYER_RELATIVE]

        army_selected = np.array([1]) if 1 in obs.observation['screen'][_SELECT] else np.array([0])

        legal_actions = obs.observation['available_actions']

        feed_dict = {self.input: [current_state],
                     self.army_selected: [army_selected]}

        output = self.session.run(self.output, feed_dict)[0]
        out = self.redistribute(output, legal_actions)

        try:
            action = np.argmax(np.random.multinomial(1, out))
        except ValueError:
            action = np.argmax(np.random.multinomial(1, out / (1 + 1e-6)))
        action = int(action)

        # The group of actions to take
        if available_actions[action] == _NO_OP:
            return actions.FunctionCall(_NO_OP, [])
        elif available_actions[action] == _SELECT_ARMY:
            return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
        elif available_actions[action] == _ATTACK_SCREEN \
                or available_actions[action] == _MOVE_SCREEN \
                or available_actions[action] == _PATROL_SCREEN \
                or available_actions[action] == _SMART_SCREEN:
            # This is the scripted one
            neutral_y, neutral_x = (screen_player_relative == _PLAYER_NEUTRAL).nonzero()
            target = [int(neutral_x.mean()), int(neutral_y.mean())]
            return actions.FunctionCall(available_actions[action], [_NOT_QUEUED, target])
        elif available_actions[action] == _STOP_QUICK:
            return actions.FunctionCall(available_actions[action], [_NOT_QUEUED])
        elif available_actions[action] == _HOLD_POSITION_QUICK:
            return actions.FunctionCall(available_actions[action], [_NOT_QUEUED])
        elif available_actions[action] == _ATTACK_MINIMAP \
                or available_actions[action] == _MOVE_MINIMAP \
                or available_actions[action] == _PATROL_MINIMAP \
                or available_actions[action] == _SMART_MINIMAP:
            neutral_y, neutral_x = (mm_player_relative == _PLAYER_NEUTRAL).nonzero()
            target = [int(neutral_x.mean()), int(neutral_y.mean())]
            return actions.FunctionCall(available_actions[action], [_NOT_QUEUED, target])
        else:
            return actions.FunctionCall(_NO_OP, [])

    def batch_predict(self, states, mini_map, army):
        return self.session.run(self.critic_output, feed_dict={self.input: states, self.minimap_input: mini_map,
                                                               self.army_selected: army})

    def predict(self, state, mini_map, army):
        return self.session.run(self.critic_output, feed_dict={self.input: [state], self.minimap_input: [mini_map],
                                                               self.army_selected: [army]})[0]

    def save(self):
        self.saver.save(self.session, '/home/rob/Documents/uni/fyp/sc2/ac_model.ckpt')

    # Defines if we have the potential of picking an illegal action and how we redistribute the probabilities
    @staticmethod
    def redistribute(output, legal_actions):
        for i, action in enumerate(available_actions):
            if action not in legal_actions:
                output[i] = 0
        if sum(output) == 0:
            for i, a in enumerate(available_actions):
                if a in legal_actions:
                    output[i] = float(1/len(legal_actions))
        else:
            output /= sum(output)
        return output


class ReplayMemory:
    def __init__(self, batch_size, max_memory_size):
        self.batch_size = batch_size
        # self.memory = []
        self.states = []
        self.minimap_states = []
        self.army_selected = []
        self.actions = []
        self.advantages = []
        self.done = []

    def add(self, state, action, reward, done):
        max_time_steps = 40
        self.states = state[0][:max_time_steps]
        self.minimap_states = state[1][:max_time_steps]
        self.army_selected = state[2][:max_time_steps]
        self.actions = action[:max_time_steps]
        self.advantages = reward[:max_time_steps]
        self.done = done[:max_time_steps]
        self.done[-1] = True

    # Update the memory to include the next state
    # def update(self, next_state, next_army, done):
    #     if len(self.memory) > 0 and not self.memory[-1][4]:
    #         self.memory[-1][4] = done
    #         self.memory[-1][5] = next_state
    #         self.memory[-1][6] = next_army

    def get_batch(self):
        # self.memory.reverse()
        # running_add = 0
        # for m, mem in enumerate(self.memory):
        #     _, _, _, r, d, _, _ = mem
        #     running_add = running_add * 0.99 + r
        #     mem[3] = running_add
        # self.memory.reverse()
        # self.advantages = (self.advantages - np.mean(self.advantages)) // (np.std(self.advantages) + 1e-10)
        return self.states, self.minimap_states, self.army_selected, self.actions, self.advantages, self.done

    def delete(self):
        del self.states[:]
        del self.minimap_states[:]
        del self.army_selected[:]
        del self.actions[:]
        self.advantages = []
        del self.done[:]

