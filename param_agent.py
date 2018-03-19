from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from DQN_PARAM import *
import matplotlib.pyplot as plt
from collections import deque
import random

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
        self.batch_size = 50
        self.max_memory_size = 5000
        self.gamma = 1
        self.learning_rate = 1e-4
        self.epsilon = 1.
        self.final_epsilon = .05
        self.epsilon_decay = 0.9999
        self.total_rewards = deque(maxlen=100)
        self.current_reward = 0

        self.allow_pick = True
        self.action = 0

        self.memory = ReplayMemory(self.num_actions, self.batch_size, self.max_memory_size, self.gamma)
        self.model = Model(self.wh, self.input_flat, self.num_actions, self.learning_rate, self.memory)
        if self.model.loaded_model:
            self.epsilon = 0.05
    """An agent specifically for solving the MoveToBeacon map."""

    def step(self, obs):

        player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
        current_state = player_relative.flatten()

        super(MoveToBeacon, self).step(obs)

        if _MOVE_SCREEN in obs.observation["available_actions"]:

            player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
            neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()

            if self.steps > 2:
                if random.random() < self.epsilon:
                    self.action = random.randint(0, self.num_actions - 1)
                    self.allow_pick = False
                else:
                    feed_dict = {self.model.screen_input: [current_state]}
                    output = self.model.session.run(self.model.output, feed_dict)[0]
                    self.action = np.argmax(output)
                    self.allow_pick = False
                target_x = self.action // 84
                target_y = self.action % 84
            else:
                target_x = neutral_x.mean()
                target_y = neutral_y.mean()
                self.action = int(target_x * 84 + target_y)

            target = [target_x, target_y]
            print(neutral_x.mean(), neutral_y.mean())
            print(target)

            reward = obs.reward
            done = False
            if reward == 1:
                done = True
                self.allow_pick = True

            self.current_reward += reward
            if obs.last():
                self.allow_pick = True
                self.total_rewards.append(self.current_reward)
                self.current_reward = 0
                if self.episodes % 100 == 0 and self.episodes > 0:
                    self.model.save()
                    print('Highest: {} | Lowest: {} | Average: {} | Timesteps: {}'.format(
                        max(self.total_rewards),
                        min(self.total_rewards),
                        np.mean(self.total_rewards),
                        self.steps)
                    )
                # if self.episodes % 1000 == 0 and self.episodes > 0:
                #     plt.plot(self.total_rewards)
                #     plt.show()

            if not neutral_y.any():
                return actions.FunctionCall(_NO_OP, [])

            if self.epsilon > self.final_epsilon:
                self.epsilon = self.epsilon * self.epsilon_decay

            self.memory.add(current_state, self.action, reward, done)
            self.model.train()

            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
        else:
            return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
