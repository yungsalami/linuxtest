from project_a5.random import LCG
import numpy as np
import matplotlib.pyplot as plt

c = 3
m = 1024
k = 30

T = np.zeros(k)

for a in range(0, k):
    lcg = LCG(0, a, c, m)
    temp = lcg.random_raw(m)
    count = np.bincount(temp.astype(int))
    T[a] = len(count[count == np.max(count)])

plt.plot(range(0, k), T, label='Period length')
plt.legend()
plt.grid(ls='dotted')
plt.xlabel("a")
plt.ylabel("T")
plt.savefig('plot_periodlength.pdf')

plt.clf()