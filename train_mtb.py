from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from DQN import Model
from DQN import ReplayMemory

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]


class TrainAgent(base_agent.BaseAgent):
    """An agent specifically for solving the MoveToBeacon map."""

    def step(self, obs):
        super(TrainAgent, self).step(obs)
        # Keep this so that it always selects the marine at the start
        if _MOVE_SCREEN in obs.observation["available_actions"]:
            player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
            # Keep this just for now, so that we can do a noop when there isn't a beacon
            neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
            if not neutral_y.any():
                return actions.FunctionCall(_NO_OP, [])
            # Just start with going to the beacon if there is one
            # target = self.get_action()
            target = [int(neutral_x.mean()), int(neutral_y.mean())]
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
        else:
            return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])

    def get_action(self, obs):
        gamma = 0.97
        learning_rate = 1e-3
        epsilon = 1.
        final_epsilon = .05
        epsilon_decay = .999

        # Memory parameters
        batch_size = 50
        max_memory_size = 10000

        memory = ReplayMemory(self.batch_size, self.max_memory_size, self.gamma)
        model = Model(self.learning_rate, self.memory)

        max_episodes = 10000
        render = True

        current_state = obs.observation["screen"][_PLAYER_RELATIVE]
        current_state = current_state.flatten()
        done = False
        count = 0
        total_reward = 0

        action = Model.get_action(current_state, self.epsilon)
        next_state, reward, done, _ = self.env.step(action)

        self.memory.add(current_state, action, reward, done, next_state)
        current_state = next_state
        total_reward += reward
        self.model.train()
        count += 1
        print('TRAIN: The episode ' + str(i) + ' lasted for ' + str(
            count) + ' time steps with epsilon ' + str(self.epsilon))

        # if i > 200:
        if self.epsilon > self.final_epsilon:
            self.epsilon *= self.epsilon_decay


agent = Agent()
agent.run()