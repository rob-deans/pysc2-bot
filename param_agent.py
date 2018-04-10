from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ac_param import *
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_SELECT = features.SCREEN_FEATURES.selected.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]


class MoveToBeacon(base_agent.BaseAgent):

    def __init__(self):
        super(MoveToBeacon, self).__init__()
        self.num_actions = 84*84
        self.input_flat = 84*84  # Size of the screen
        self.wh = 84

        self.batch_size = 60
        self.max_memory_size = 2000

        self.gamma = .99
        self.actor_lr = 1e-3
        self.critic_lr = 5e-3

        self.total_rewards = deque(maxlen=100)
        self.current_reward = 0

        self.allow_pick = True
        self.action = 0

        self.targets = []
        self.beacons = []
        self.beacons_store = True

        self.memory = ReplayMemory(self.batch_size, self.max_memory_size)
        self.model = ActorCriticModelCont(self.wh, self.input_flat, self.num_actions,
                                          self.actor_lr, self.critic_lr, self.memory, self.gamma)
    """An agent specifically for solving the MoveToBeacon map."""

    def step(self, obs):

        player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
        current_state = player_relative.flatten()

        super(MoveToBeacon, self).step(obs)

        done = False

        if _MOVE_SCREEN in obs.observation["available_actions"]:

            player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
            neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
            if self.beacons_store:
                self.beacons.append([neutral_x, neutral_y])
                self.beacons_store = False

            self.action = self.model.run(current_state)
            try:
                action_ = np.argmax(np.random.multinomial(1, self.action))
            except ValueError:
                action_ = np.argmax(np.random.multinomial(1, self.action/(1+1e-6)))

            self.allow_pick = False
            target_x = action_ // 84
            target_y = action_ % 84

            target = [target_x, target_y]
            self.targets.append(target)

            reward = obs.reward
            if reward == 1:
                print(target)
                self.beacons_store = True
                self.allow_pick = True
                # self.model.train()

            # STATS ONLY
            self.current_reward += reward
            if obs.last():
                done = True
                self.allow_pick = True
                self.total_rewards.append(self.current_reward)
                if self.episodes % 100 == 0 and self.episodes > 0:
                    self.model.save()
                    print('Highest: {} | Lowest: {} | Average: {} | Timesteps: {}'.format(
                        max(self.total_rewards),
                        min(self.total_rewards),
                        np.mean(self.total_rewards),
                        self.steps)
                    )
                if self.current_reward > 0:
                    temp = np.array(self.targets)
                    temp_2 = np.array(self.beacons)
                    plt.scatter(temp_2[:, 0], temp_2[:, 1])
                    plt.scatter(temp[:, 0], temp[:, 1])
                    plt.title('Episode {}'.format(self.episodes))
                    plt.show()
                del self.targets[:]
                del self.beacons[:]
                self.current_reward = 0

            if not neutral_y.any():
                return actions.FunctionCall(_NO_OP, [])

            actions_oh = np.zeros(7056)
            actions_oh[action_] = 1
            self.memory.add(current_state, actions_oh, reward, done)
            self.model.train()

            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
        else:
            return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
