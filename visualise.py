import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.animation as animation
import numpy as np

actions = pickle.load(open('policy_actions.pkl', 'rb'))
rewards = pickle.load(open('policy_rewards.pkl', 'rb'))

fig = plt.figure()
# ax1 = fig.add_subplot(1, 1, 1)
# count = 0
#
#
# def animate(i):
#     ax1.clear()
#     ax1.plot(rewards[:i])
#
#
# ani = animation.FuncAnimation(fig, animate, interval=50, repeat=False)
# # plt.plot(rewards)
# plt.show()

ax2 = fig.add_subplot(1, 1, 1)
ax2.set_xlim(0, 16)


def animate_hist(i):
    ax2.clear()
    ax2.hist(actions[:i*240])


ani = animation.FuncAnimation(fig, animate_hist, interval=50, repeat=False)
plt.show()
