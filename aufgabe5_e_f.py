from project_a5.random import LCG
import numpy as np
import matplotlib.pyplot as plt

a = 1601
c = 3456
m = 10000

# ---------- e) ----------
lcg = LCG(69, a, c, m)

lcg_data = lcg.uniform(size=1000) # create uniform distribution

x_2d, y_2d = lcg_data[::2], lcg_data[1::2] # put in groups of two

lcg_data = lcg_data[:-1]

x_3d, y_3d, z_3d = lcg_data[::3], lcg_data[1::3], lcg_data[2::3] # put in groups of three

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 2, projection='3d')

ax1.scatter(
    x_3d, y_3d, z_3d,
    s = 5,
    alpha=0.7,
) # 3d scatter plot as described on the exercise sheet

ax1.view_init(elev=25, azim=35)

ax2 = fig.add_subplot(2, 2, 1)

ax2.scatter(
    x_2d, y_2d,
    s = 5,
    alpha=0.7,
) # two dimensional scatter plot

plt.savefig('streu.pdf')

plt.clf()

# ---------- f) ----------

rand = np.random.default_rng(69) # same seed as above

u, v = rand.uniform(0, 1, size=(2, 1000)) # create 2 dim. random numbers

x, y, z = rand.uniform(0, 1, size=(3, 1000)) # create 3 dim. random numbers

ax1 = fig.add_subplot(1, 2, 2, projection='3d')

ax1.scatter(
    x, y, z,
    s = 5,
    alpha=0.7,
) # 3d scatter plot as described on the exercise sheet

ax1.view_init(elev=25, azim=35)

ax2 = fig.add_subplot(2, 2, 1)

ax2.scatter(
    u, v,
    s = 5,
    alpha=0.7,
) # 2d scatter plot

plt.savefig('streu_f.pdf')