from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from DQN import ReplayMemory
from DQN import Model

import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold='nan')

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
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
    # _MOVE_CAMERA,
    # _SELECT_POINT,
    # _SELECT_RECT,
    # _SELECT_CONTROL_GROUP,
    # _STOP_QUICK,
    # _SELECT_ARMY,
    # _ATTACK_SCREEN,
    _MOVE_SCREEN,
    # _MOVE_MINIMAP,
    # _PATROL_SCREEN,
    # _PATROL_MINIMAP,
    # _ATTACK_MINIMAP,
    # _HOLD_POSITION_QUICK,
    # _SMART_SCREEN,
    # _SMART_MINIMAP
]


class MoveToBeacon(base_agent.BaseAgent):
    """An agent specifically for solving the MoveToBeacon map."""
    def __init__(self):
        super(MoveToBeacon, self).__init__()
        self.num_actions = len(available_actions)
        self.num_input = 84 * 84  # Size of the screen
        self.batch_size = 32
        self.max_memory_size = 10000
        self.gamma = 1.
        self.learning_rate = 1e-4
        self.epsilon = 1.
        self.final_epsilon = .02
        self.epsilon_decay = 0.9999
        self.total_rewards = []
        self.current_reward = 0

        self.memory = ReplayMemory(self.num_actions, self.batch_size, self.max_memory_size, self.gamma)
        self.model = Model(self.num_input, self.num_actions, self.learning_rate, self.memory)

    def step(self, obs):

        # Current observable state
        player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
        current_state = player_relative.flatten()

        super(MoveToBeacon, self).step(obs)
        # for now it will always select the army
        if _MOVE_SCREEN in obs.observation['available_actions']:

            # Here we will attempt to choose the correct action and just the rest to script the "MOVE_SCREEN" action
            action = self.model.get_action(current_state, self.epsilon)
            # print('Action taken: {}'.format(action))
            reward = obs.reward
            done = obs.last()
            self.current_reward += reward
            if done:
                self.total_rewards.append(self.current_reward)
                print('Reward: {}'.format(self.current_reward))
                print('Epsilon: {}'.format(self.epsilon))
                self.current_reward = 0
            if self.steps > 49999:
                plt.plot(self.total_rewards)
                plt.show()

            self.memory.add(current_state, action, reward, done)

            # no op
            if action == 0:
                return actions.FunctionCall(_NO_OP, [])

            # This is the scripted one
            neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()  # Get the location of the beacon
            target = [int(neutral_x.mean()), int(neutral_y.mean())]

            if self.epsilon > self.final_epsilon:
                self.epsilon = self.epsilon * self.epsilon_decay

            return actions.FunctionCall(available_actions[action], [_NOT_QUEUED, target])
        else:
            return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
