from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from policy import ReplayMemory
from policy import Model

import random
import matplotlib.pyplot as plt
import numpy as np

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
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
    # _MOVE_CAMERA,
    # _SELECT_POINT,
    # _SELECT_RECT,
    # _SELECT_CONTROL_GROUP,
    # _STOP_QUICK,
    _SELECT_ARMY,
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

actions_num = [0, 7, 331]


class MoveToBeacon(base_agent.BaseAgent):
    """An agent specifically for solving the MoveToBeacon map."""
    def __init__(self):
        super(MoveToBeacon, self).__init__()
        self.num_actions = len(actions_num)
        self.input_flat = 84*84  # Size of the screen
        self.wh = 84
        self.batch_size = 10
        self.gamma = .99
        self.learning_rate = 1e-3
        self.run = 0

        self.actions = []
        self.states = []
        self.army_state = []

        # Stat count
        self.total_rewards = []
        self.current_reward = 0
        self.actions_taken = [0, 0, 0]
        self.rewards = []

        self.memory = ReplayMemory()
        self.model = Model(self.wh, self.input_flat, 1, self.num_actions, self.learning_rate, self.memory)

    def discount_rewards(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, len(r))):
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def process_rewards(self, reward):
        return [reward] * 240

    def step(self, obs):
        # Current observable state
        player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
        current_state = player_relative.flatten()
        army_selected = np.array([1]) if 1 in obs.observation['screen'][_SELECT] else np.array([0])

        super(MoveToBeacon, self).step(obs)

        if obs.first():
            del self.states[:]
            del self.actions[:]
            del self.army_state[:]
            del self.rewards[:]

        legal_actions = obs.observation['available_actions']

        feed_dict = {self.model.screen_input: [current_state], self.model.army_input: [army_selected]}

        output = self.model.session.run(self.model.output, feed_dict)[0]
        out = redistribute(output, legal_actions)
        action = int(np.argmax(np.random.multinomial(1, out / (1 + 1e-6))))

        self.actions_taken[int(action)] += 1

        self.states.append(current_state)
        self.army_state.append(army_selected)
        actions_oh = np.zeros(3)
        actions_oh[action] = 1
        self.actions.append(actions_oh)

        # reward = obs.reward
        self.rewards.append(obs.reward)
        self.current_reward += obs.reward
        if obs.reward == 1 or obs.last():

            # if self.current_reward == 0:
            #     reward = -1

            # rewards = [reward] * len(self.actions)
            rewards_discounted = self.discount_rewards(self.rewards)
            self.memory.add(self.states, self.army_state, self.actions, rewards_discounted)
            # Delete all the actions and states ready for more to be appended
            del self.states[:]
            del self.actions[:]
            del self.army_state[:]
            del self.rewards[:]
            if self.episodes % self.batch_size == 0 and self.episodes > 0:
                self.model.train()

        if obs.last():

            # Printing out the stats
            self.total_rewards.append(self.current_reward)
            self.current_reward = 0
            if self.episodes % 100 == 0 and self.episodes > 0:
                print('Highest: {} | Lowest: {} | Average: {} | Time steps: {}'.format(
                        max(self.total_rewards[-100:]),
                        min(self.total_rewards[-100:]),
                        np.mean(self.total_rewards[-100:]),
                        self.steps
                    )
                )
                print(self.actions_taken)
            if self.episodes % 1000 == 0 and self.episodes > 0:
                plt.plot(self.total_rewards)
                plt.show()
        # End stats #

        # The group of actions to take
        if available_actions[action] == _NO_OP:
            return actions.FunctionCall(_NO_OP, [])
        elif available_actions[action] == _SELECT_ARMY:
            return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
        elif available_actions[action] == _MOVE_SCREEN:
            # This is the scripted one
            neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()  # Get the location of the beacon
            target = [int(neutral_x.mean()), int(neutral_y.mean())]
            return actions.FunctionCall(available_actions[action], [_NOT_QUEUED, target])


# Defines if we have the potential of picking an illegal action and how we redistribute the probabilities
def redistribute(output, legal_actions):
    if 331 not in legal_actions and output[2] != 0:
        output[2] = 0
        output[:2] /= sum(output)
    return output


