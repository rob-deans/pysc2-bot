from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from policy_param import ReplayMemory
from policy_param import Model
import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
import pickle

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
        self.wh = 64
        self.num_actions = self.wh ** 2
        self.input_flat = self.wh ** 2  # Size of the screen

        self.batch_size = 10
        self.gamma = .99
        self.learning_rate = 1e-3

        self.actions = []
        self.states = []
        self.rewards = []

        # Stat count
        self.total_rewards = []
        self.total_actions = []
        self.current_reward = 0
        self.actions_taken = np.zeros(self.num_actions)

        self.memory = ReplayMemory()
        self.model = Model(self.input_flat, self.num_actions, self.learning_rate, self.memory)
    """An agent specifically for solving the MoveToBeacon map."""

    def step(self, obs):

        player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
        current_state = player_relative.flatten()
        current_state = [1 if c == 3 else 0 for c in current_state]

        super(MoveToBeacon, self).step(obs)

        if obs.first():
            del self.states[:]
            del self.actions[:]
            del self.rewards[:]

        if _MOVE_SCREEN in obs.observation["available_actions"]:

            player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
            neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
            if not neutral_y.any():
                return actions.FunctionCall(_NO_OP, [])

            feed_dict = {self.model.screen_input: [current_state]}

            output = self.model.session.run(self.model.output, feed_dict)[0]
            try:
                action_ = np.argmax(np.random.multinomial(1, output))
            except ValueError:
                action_ = np.argmax(np.random.multinomial(1, output/(1+1e-6)))

            target_x = action_ // 64
            target_y = action_ % 64

            target = [target_x, target_y]
            self.targets.append(target)

            self.states.append(current_state)
            actions_oh = np.zeros(self.num_actions)
            actions_oh[action_] = 1
            self.actions.append(actions_oh)

            reward = obs.reward
            self.rewards.append(reward)

            if reward == 1:
                print(target)

            if obs.last():
                rewards_discounted = self.discount_rewards(self.rewards)
                self.memory.add(self.states, self.actions, rewards_discounted)
                # Delete all the actions and states ready for more to be appended
                del self.states[:]
                del self.actions[:]
                del self.rewards[:]
                if self.episodes % self.batch_size == 0 and self.episodes > 0:
                    self.model.train()

            # STATS ONLY
            self.current_reward += reward
            if obs.last():
                self.total_rewards.append(self.current_reward)
                self.log_rewards.append(self.current_reward)
                if self.episodes % 100 == 0 and self.episodes > 0:
                    self.model.save()
                    pickle.dump(self.log_rewards, open('./policy_results.pkl', 'wb'))
                    print('Highest: {} | Lowest: {} | Average: {} | Timesteps: {}'.format(
                        max(self.total_rewards),
                        min(self.total_rewards),
                        np.mean(self.total_rewards),
                        self.steps)
                    )
                self.current_reward = 0

            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
        else:
            return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])

    def discount_rewards(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r, dtype=float)
        running_add = 0
        for t in reversed(range(0, len(r))):
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r
