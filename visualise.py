import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.animation as animation
import numpy as np
#
# actions = pickle.load(open('actions.pkl', 'rb'))
# rewards1 = pickle.load(open('policy_rewards1.pkl', 'rb'))
# rewards2 = pickle.load(open('policy_rewards2.pkl', 'rb'))
# rewards3 = pickle.load(open('policy_rewards3.pkl', 'rb'))
# rewards4 = pickle.load(open('policy_rewards4.pkl', 'rb'))
# rewards5 = pickle.load(open('policy_rewards5.pkl', 'rb'))
_rewards = list(pickle.load(open('rewards4.pkl', 'rb')))
# _rewards1 = list(pickle.load(open('rewards.pkl1', 'rb')))
# _rewards2 = list(pickle.load(open('rewards2.pkl', 'rb')))
# _rewards3 = list(pickle.load(open('rewards.pkl', 'rb')))

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
count = 0


def animate(i):
    ax1.clear()
    ax1.plot(_rewards[:i])
    # ax1.plot(rewards2[:i])
    # ax1.plot(rewards3[:i])
    # ax1.plot(rewards4[:i])
    # ax1.plot(rewards5[:i])


ani = animation.FuncAnimation(fig, animate, interval=50, repeat=False)
# plt.plot(rewards)
plt.show()
# #
# ax2 = fig.add_subplot(1, 1, 1)
# ax2.set_xlim(0, 16)
#
#
# def animate_hist(i):
#     ax2.clear()
#     ax2.hist(actions[:i*240])
#
#
# ani = animation.FuncAnimation(fig, animate_hist, interval=50, repeat=False)
# plt.show()
