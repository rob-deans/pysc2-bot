from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from actor_critic import ReplayMemory
from actor_critic import ActorCriticModel

import numpy as np
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
        # Screen sizes
        self.input_flat = 84*84  # Size of the screen
        self.wh = 84
        # Minimap sizes
        self.mm_input_flat = 64*64
        self.mm_wh = 64

        self.batch_size = 32
        self.max_memory_size = 2000

        self.gamma = .99
        self.actor_lr = 1e-3
        self.critic_lr = 5e-3

        self.actions = []
        self.states = []
        self.minimap_states = []
        self.army_state = []
        self.done = []

        # Stat count
        self.total_rewards = deque(maxlen=100)
        self.episode_reward = 0
        self.actions_taken = np.zeros(self.num_actions)

        self.memory = ReplayMemory(self.batch_size, self.max_memory_size)
        self.model = ActorCriticModel(self.wh, self.input_flat, self.mm_wh, self.mm_input_flat,
                                      self.num_actions, self.actor_lr, self.critic_lr,
                                      self.memory, self.gamma)

    def step(self, obs):
        # Current observable state
        screen_relative = obs.observation["screen"][_PLAYER_RELATIVE]
        current_state = screen_relative.flatten()
        mini_map_relative = obs.observation['minimap'][_MM_PLAYER_RELATIVE]
        minimap_state = mini_map_relative.flatten()
        army_state = obs.observation['screen'][_SELECT].flatten()

        if len(self.memory.memory) > 0:
            self.memory.update(army_state)
            self.model.train()

        super(MoveToBeacon, self).step(obs)

        legal_actions = obs.observation['available_actions']

        feed_dict = {self.model.army_selected: [army_state]}

        output = self.model.session.run(self.model.output, feed_dict)[0]
        out = redistribute(output, legal_actions)
        try:
            action = int(np.argmax(np.random.multinomial(1, out)))
        except ValueError:
            action = int(np.argmax(np.random.multinomial(1, out / (1 + 1e-6))))

        self.actions_taken[int(action)] += 1

        self.states.append(current_state)
        self.minimap_states.append(minimap_state)
        self.army_state.append(army_state)

        actions_oh = np.zeros(self.num_actions)
        actions_oh[action] = 1
        self.actions.append(actions_oh)

        reward = obs.reward
        self.episode_reward += reward

        if obs.last():
            # if self.episode_reward == 0:
            #     reward = -100
            # Printing out the stats
            self.total_rewards.append(self.episode_reward)
            self.episode_reward = 0
            if self.episodes % 100 == 0 and self.episodes > 0:
                self.model.save()
                print('Highest: {} | Lowest: {} | Average: {} | Time steps: {}'.format(
                        max(self.total_rewards),
                        min(self.total_rewards),
                        np.mean(self.total_rewards),
                        self.steps
                    )
                )
                print(self.actions_taken)

        self.memory.add(army_state, actions_oh, reward, obs.last())

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
            neutral_y, neutral_x = (screen_relative == _PLAYER_NEUTRAL).nonzero()
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
            neutral_y, neutral_x = (mini_map_relative == _PLAYER_NEUTRAL).nonzero()
            target = [int(neutral_x.mean()), int(neutral_y.mean())]
            return actions.FunctionCall(available_actions[action], [_NOT_QUEUED, target])
        else:
            return actions.FunctionCall(_NO_OP, [])


# Defines if we have the potential of picking an illegal action and how we redistribute the probabilities
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


