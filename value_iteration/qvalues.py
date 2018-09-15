import numpy as np
import matplotlib.pyplot as plt
from dart import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

dart_info = DartInfo()
qvalues = np.load('qvalues.npy')

actions = np.empty(shape=(302, 302, 3)) #cannot use dtype=int due to nan
actions.fill(np.nan)

for i in range(2,302):
    for j in range(2, i+1):
        for k in range(0,3):
            if j >= i - k * 60:
                q_ijk = qvalues[i, j, k, :]
                idx = np.where(q_ijk == np.nanmax(q_ijk))[0]
                actions[i,j,k] = idx[0]


def map_color(a):
    if dart_info.is_outer_single(a):
        return 'r'
    if dart_info.is_inner_single(a):
        return 'b'
    if dart_info.is_treble(a):
        return 'g'
    if dart_info.is_double(a):
        return 'c'
    if a == 80:
        return 'm'
    if a == 81:
        return 'k'


def map_score(a):
    if not np.isnan(a):
        return SCORE[int(a)]
    else:
        return np.nan


colors_0 = np.vectorize(map_color)(actions[:, :, 0]).ravel()
scores_0 = np.vectorize(map_score)(actions[:, :, 0])
colors_1 = np.vectorize(map_color)(actions[:, :, 1]).ravel()
scores_1 = np.vectorize(map_score)(actions[:, :, 1])
colors_2 = np.vectorize(map_color)(actions[:, :, 2]).ravel()
scores_2 = np.vectorize(map_score)(actions[:, :, 2])

fig = plt.figure()
# Meshgrid definition (x,y):
# (0,0),(1,0),(2,0)...
# (0,1),(1,1),(2,1)...
# --> y fixed, x varies, "swapped" roles of axes
y, x = np.meshgrid(np.arange(0,302),np.arange(0,302))

ax0 = fig.add_subplot(221, projection='3d')
ax0.scatter(xs=x, ys=y, zs=scores_0, c=colors_0, marker='.', linewidths=1)
ax1 = fig.add_subplot(222, projection='3d')
ax1.scatter(xs=x, ys=y, zs=scores_1, c=colors_1, marker='.', linewidths=1)
ax2 = fig.add_subplot(223, projection='3d')
ax2.scatter(xs=x, ys=y, zs=scores_2, c=colors_2, marker='.', linewidths=1)
ax = fig.add_subplot(224, projection='3d')


legend_elements = [
    Line2D([0], [0], color='r', lw=2, label='Outer Single'),
    Line2D([0], [0], color='b', lw=2, label='Inner Single'),
    Line2D([0], [0], color='g', lw=2, label='Treble'),
    Line2D([0], [0], color='c', lw=2, label='Double'),
    Line2D([0], [0], color='m', lw=2, label='Outer Bullseye'),
    Line2D([0], [0], color='k', lw=2, label='Bullseye')
]


ax0.set_xlabel('Initial score $s_0$')
ax0.set_ylabel('Score before 1st attempt $s_0$')
ax0.set_zlabel('Action score')

ax1.set_xlabel('Initial score $s_0$')
ax1.set_ylabel('Score before 2nd attempt $s_1$')
ax1.set_zlabel('Action score')

ax2.set_xlabel('Initial score $s_0$')
ax2.set_ylabel('Score before 3rd attempt $s_2$')
ax2.set_zlabel('Action score')

ax.axis('off')
ax.legend(handles=legend_elements, loc='center')

plt.show()