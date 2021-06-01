import numpy as np 
import pandas
import matplotlib.pyplot as plt


# ---------- a) ----------

# create data frame
data_neutrinos = pandas.DataFrame()

# set constants
tev = 10**(12)
gamma = 2.7
N = 10**5
seed = 8787

# draw uniform random numbers
rng = np.random.default_rng(seed)
data = rng.uniform(low = 0, high = 1, size = N)

# define neutrino flux distribution

def f(y):
    return (1 - y)**(1/(1- gamma))

events = f(data)

data_neutrinos['Energy'] = events


# --------- b) ---------

def p(y):
    return (1 - np.exp(-y/2))**3

u = rng.uniform(0, 1, N)
neumann_mask = p(events) > u

detection = events[neumann_mask]

data_neutrinos['AcceptanceMask'] = neumann_mask

# make plots
plt.figure('b) Stuffs')
plt.hist(events, bins=N, histtype="step", label = "Events")
plt.hist(detection, bins=N, histtype="step", label = "Detected events")

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Energy")
plt.ylabel("Neutrino flux")

plt.legend()
plt.savefig('neutrinos.pdf')


# ---------- d) ----------

# import Generator class
from project_a5.random import Generator

gen = Generator(seed = seed)

hits_number = np.empty(N)

i = 0 # counter variable
while (i < N):
    hit = int(gen.normal(loc = 10*events[i], scale = 2*events[i], size = 1)) # get random value, turn it into integer
    if hit > 0: # only save if it is greater than 0
        hits_number[i] = hit
        i += 1

data_neutrinos['NumberOfHits'] = hits_number


# ---------- e) ----------

x_loc = 7
y_loc = 3
sigma = 1/(np.log10(hits_number + 1)) # set standard deviation

def location(sigma, loc):
    x = gen.normal(loc = loc, scale = sigma, size = len(sigma)) # get random numbers

    x_inside = np.logical_or(x < 0, x >= 10) # mask to test whether it is contained within the detector limits

    if len(x[x_inside]) < len(sigma): # see if values have been excluded
        x = np.r_[x[x_inside], location(sigma[x_inside], loc)] # for the excluded values get new ones

    return x

# do this for x and y
x = location(sigma, x_loc)
y = location(sigma, y_loc)

#plt.figure('e) Ortsmessung')
#plt.hist2d(x, y, bins=(100, 100), range=[[0, 10], [0, 10]])
#plt.colorbar()
#
#plt.xlabel('x')
#plt.ylabel('y')
#plt.savefig('ort.pdf')
#

# Unfortunately when we try to plot the data there is a problem with the binning, it says that the bin dimensions must fit the dimensions of the input data.
# However, we have used the same exact code to create the 2d-Histogram in f) and it worked there, strangely. 


# ---------- f) ----------

# set new parameters 
N_f = 10**7
sigma_f_loc = 3
rho = 0.5
mu = 5

hits_f_log = gen.normal(2, 1, N_f) # normal distribution for hits
hits_f = np.round(10**hits_f_log) # get non logarithmic hits

# calculate location as shown on the exercise sheet (with detector center x = 5, y = 5)
x_ = gen.standard_normal(size = N_f)
y_ = gen.standard_normal(size = N_f)

x_f = np.sqrt(1 - rho**2) * sigma_f_loc * x_ + rho * sigma_f_loc * y_ + mu
y_f = sigma_f_loc * y_ + mu

# plot logarithmic hits
plt.figure('f) Untergrund MC')
plt.hist(hits_f_log, bins=100, density=True, label='Logarithmus der Anzahl der Hits')
plt.legend()
plt.savefig('unter.pdf')

# plot location distribution
plt.figure('f) Untergrund Ortsmessung')
plt.hist2d(x_f, y_f, bins=(100, 100), range=[[0, 10], [0, 10]])
plt.colorbar()

plt.xlabel('x')
plt.ylabel('y')
plt.savefig('unterort.pdf')

data_background = pandas.DataFrame()

data_background['NumberOfHits'] = hits_f
data_background['x'] = x_f
data_background['y'] = y_f 

# convert to hdf
data_neutrinos.to_hdf("NeutrinoMC.hdf5", "Signal")
data_background.to_hdf("NeutrinoMC.hdf5", "Background")