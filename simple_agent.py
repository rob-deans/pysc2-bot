from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from DQN import ReplayMemory
from DQN import Model

import random
import numpy as np
import pickle
from collections import deque

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


class MoveToBeacon(base_agent.BaseAgent):
    """An agent specifically for solving the MoveToBeacon map."""
    def __init__(self):
        super(MoveToBeacon, self).__init__()

        self.num_actions = len(available_actions)
        self.input_flat = 84*84  # Size of the screen
        self.wh = 84
        # Minimap sizes
        self.mm_input_flat = 64*64
        self.mm_wh = 64

        self.batch_size = 32
        self.max_memory_size = 5000

        self.gamma = .99
        self.learning_rate = 1e-4
        self.epsilon = 1.
        self.final_epsilon = .05
        self.epsilon_decay = 0.999

        self.total_rewards = deque(maxlen=100)
        self.current_reward = 0
        self.actions_taken = np.zeros(self.num_actions)
        self.rewards = []

        self.total_actions = []

        self.memory = ReplayMemory(self.num_actions, self.batch_size, self.max_memory_size, self.gamma)
        self.model = Model(self.wh, self.input_flat, self.mm_wh, self.mm_input_flat,
                           1, self.num_actions, self.learning_rate, self.memory)
        if self.model.loaded_model:
            self.epsilon = 0.05

    def step(self, obs):
        # Current observable state
        screen_player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
        current_state = screen_player_relative.flatten()
        mm_player_relative = obs.observation['minimap'][_MM_PLAYER_RELATIVE]
        minimap_state = mm_player_relative.flatten()

        army_selected = np.array([1]) if 1 in obs.observation['screen'][_SELECT] else np.array([0])

        super(MoveToBeacon, self).step(obs)

        legal_actions = obs.observation['available_actions']

        if random.random() < self.epsilon:
            action = legal_actions[random.randint(0, len(legal_actions)) - 1]
            action = available_actions.index(action)
        else:
            feed_dict = {self.model.screen_input: [current_state], self.model.minimap_input: [minimap_state],
                         self.model.army_input: [army_selected]}
            output = self.model.session.run(self.model.output, feed_dict)[0]
            output = [value if action in legal_actions else -9e10 for action, value in zip(available_actions, output)]
            action = np.argmax(output)
            self.actions_taken[int(action)] += 1
        self.total_actions.append(action)

        # print('Action taken: {}'.format(action))
        reward = obs.reward
        done = False
        if reward == 1:
            done = True

        self.current_reward += reward
        if obs.last():
            self.total_rewards.append(self.current_reward)
            self.rewards.append(self.current_reward)
            self.current_reward = 0
            if self.episodes % 100 == 0 and self.episodes > 0:
                self.model.save()
                print('Highest: {} | Lowest: {} | Average: {}'.format(
                    max(self.total_rewards),
                    min(self.total_rewards),
                    np.mean(self.total_rewards))
                )
                print(self.actions_taken)
            if self.episodes % 1500 == 0 and self.episodes > 0:
                pickle.dump(self.total_actions, open('/home/rob/Documents/uni/fyp/sc2/actions4.pkl', 'wb'))
                pickle.dump(self.rewards, open('/home/rob/Documents/uni/fyp/sc2/rewards4.pkl', 'wb'))
                exit(0)

        if self.epsilon > self.final_epsilon:
            self.epsilon = self.epsilon * self.epsilon_decay

        self.memory.add([current_state, minimap_state, army_selected], action, reward, done)
        self.model.train()

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
