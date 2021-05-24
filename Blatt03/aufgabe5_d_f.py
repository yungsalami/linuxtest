from project_a5.random import LCG
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------- d) ----------
a = 1601
c = 3456
m = 10000
k = 9 
df = pd.DataFrame()

for seed in range(k):
    lcg = LCG(seed, a, c, m)
    temp = "seed=" + str(seed)
    df[temp] = lcg.uniform(size=m)

df.hist(alpha=0.8, bins=26, figsize=(16,16))
plt.savefig('hist.pdf')

# --------- noch zu f) ----------

df_f = pd.DataFrame()

for seed in range(k):
    rand = np.random.default_rng(seed)
    rn = rand.uniform(0, 1, m)
    temp = "seed=" + str(seed)
    df_f[temp] = rn

df_f.hist(alpha=0.8, bins=26, figsize=(16,16))
plt.savefig('hist_f.pdf')