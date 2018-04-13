from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
from pysc2.env import sc2_env
from absl import flags

from actor_critic import ActorCriticModel

FLAGS = flags.FLAGS


def run_loop(agent, env, max_frames=60):
    """A run loop to have agents and an environment interact."""
    start_time = time.time()
    current_state = env.reset()
    rollout = []
    try:
        for i in range(max_frames):
            # Take an action
            action = agent.get_action(current_state)
            next_state, reward, done = env.step(action)
            rollout.append([current_state, action, reward, done, next_state])
            current_state = next_state
        yield rollout
    except KeyboardInterrupt as e:
        print(e)
    finally:
        elapsed_time = time.time() - start_time
        print("Took %.3f seconds" % elapsed_time)


def collect_rollout(agent, env):
    rollout = run_loop(agent, env, 60)
    # agent.train(rollout)


def train(agent, env):
    for i in range(1000):
        collect_rollout(agent, env)


if __name__ == '__main__':
    FLAGS(sys.argv)
    sc_env = sc2_env.SC2Env(
        map_name="MoveToBeacon",
        visualize=False,
        screen_size_px=(84, 84),
        minimap_size_px=(64, 64),
    )
    # --agent
    # simple_agent.MoveToBeacon - -map
    # MoveToBeacon - -max_agent_steps = 1000000 - -norender

    num_actions = 16
    # Screen sizes
    input_flat = 84 * 84  # Size of the screen
    wh = 84
    # Minimap sizes
    mm_input_flat = 64 * 64
    mm_wh = 64

    batch_size = 8
    max_memory_size = 2000

    gamma = .99
    actor_lr = 1e-3
    critic_lr = 5e-3

    ac_agent = ActorCriticModel(wh, input_flat, num_actions, actor_lr, critic_lr, gamma)

    train(ac_agent, sc_env)

