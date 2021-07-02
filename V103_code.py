# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 10:42:30 2021

@author: salem
"""
import uncertainties 
import math as m
from scipy.optimize import curve_fit
from scipy.signal import argrelmin
from scipy.signal import find_peaks
import os
import pandas as pd
import numpy as np
import matplotlib as plt
import csv
import matplotlib.pyplot as plt
# Daten ziehen

# =============================================================================
# =============================================================================
# # #   Daten einspeisen
# =============================================================================
# =============================================================================


class stab:
    def __init__( self, no,  data, gewichte, masse, length, area):
        self.name = no
        self.data = data
        



def latex(a):
    print(a.to_latex(decimal = ",", index=False))



## Daten

pa = np.arange(5,50,5)*10**(-2)
pa = np.append(pa,49*10**(-2))


#linke Seite beim beidseitigem

pbl = np.array([50.5, 45.5, 40.5, 35.5, 30.5])*10**(-2)
pbr = np.array([5, 10, 15, 20, 25])*10**(-2)

## Stab: stab# = [einseitig, links, rechts]
    
stab1a = np.array([ 130, 430, 870, 1450, 2120, 2870, 3715, 4630, 5630, 6180])*10**(-6)

stab1b = np.array( [ [ 210, 400, 540, 650, 690], 
                    [ 60, 190, 355, 505, 640] ])*10**(-6)
                  

# eckig, silber
stab1_length = 59.1*10**(-2)            # m
stab1_area = 3.61*10**(-4)              # m^2
stab1_volume = stab1_length*stab1_area	# m^3
stab1_weight = 0.1634                      # kg

stab1_mass = [ 0.85, 2.357]     # [einseitig, zweiseitig]

##################################

stab2a = np.array([ 70, 90, 380, 645, 920, 1235, 1530, 2010, 2400, 2785])*10**(-6)

stab2b = np.array([[ 130, 250, 345, 405, 430],
                 [ 40, 120, 230, 330, 400]])*10**(-6)
                 

# eckig, braun
stab2_length = 60.3*10**(-2)            # m
stab2_area = 1.15*10**(-4)              # m^2
stab2_volume = stab2_length*stab2_area	# m^3
stab2_weight = 0.604                      # kg
stab2_mass = np.array([0.850, 3.515])


################################


stab3a = np.array([50, 160, 340, 555, 820, 1140, 1500, 1940, 2245, 2455])*10**(-6)
stab3b = np.array([[ 270, 520, 730, 870, 945],
                   [90, 260, 490, 710, 860]])*10**(-6)


# rund, golden
stab3_length = 60*10**(-2)            # m
stab3_area = 0.5*0.5*np.pi*10**(-4)       # m^2
stab3_volume = stab3_length*stab3_area	# m^3
stab3_weight = 0.393                      # kg
stab3_mass = np.array([0.25, 2.357])
           
###############################

stab4a = np.array([60, 155, 340, 530, 830, 1125, 1470, 1850, 2115, 2510])*10**(-6)
stab4b = np.array([[160, 300, 440, 545, 600],
                   [60, 155, 295, 440, 530]])*10**(-6)


# braun, rund
stab4_length = 60*10**(-2)            # m
stab4_area = 0.5*0.5*np.pi*10**(-4)       # m^2
stab4_volume = stab4_length*stab4_area	# m^3
stab4_weight = 0.364                      # kg
stab4_mass = np.array([0.55, 3.515])


# =============================================================================
# plots
# =============================================================================


### stab 1

a = ((0.5*10**(-2))**4)*(np.pi/4)
b = ((1.9*10**(-2))**4)/12
def xaxis(L, x):    
    return L*(x**2) - (x**3)/3



# darf man ohne weiteres annehmen, dass die neutrale Phase die geometrische Mitte ist??


I = np.array([1.086, 1.086, 0.049, 0.049 ])*10**(-8) # stab1, stab2,... ; keine richtigen Daten für stab2


def E(m, mass, I ):
    F = 9.81*mass
    dE = abs(F/(2*I*m[0]*m[0]))*m[1]
    return round(F/(2*I*m[0]), 2), round(dE,2)



# =============================================================================
# =============================================================================
# # Auswertung (einseitig)
# =============================================================================
# =============================================================================

#stab 1 mit plot
x1a = xaxis(0.52, pa)

fit1a, cov1a = np.polyfit(x1a,stab1a,1, cov=True)
poly1d_fn = np.poly1d(fit1a) 
m1a = np.array([fit1a[0], np.sqrt(cov1a[0,0])])

plt.figure()
plt.plot(x1a, stab1a, 'x')
plt.plot(x1a, poly1d_fn(x1a),'--k')

# stab 2
fit2a, cov2a = np.polyfit(x1a, stab2a, 1, cov=True)
m2a = np.array([fit2a[0], np.sqrt(cov2a[0,0])])

# stab 3
fit3a, cov3a = np.polyfit(x1a, stab3a, 1, cov=True)
m3a = np.array([fit3a[0], np.sqrt(cov3a[0,0])])

# stab 4
fit4a, cov4a = np.polyfit(x1a, stab4a, 1, cov=True)
m4a = np.array([fit4a[0], np.sqrt(cov4a[0,0])])

ma = np.array([m1a, m2a, m3a, m4a])


E1a = E(m1a, stab1_mass[0], I[0])
E2a = E(m2a, stab2_mass[0], I[1])
E3a = E(m3a, stab3_mass[0], I[2])
E4a = E(m4a, stab4_mass[0], I[3])

Ea = np.array([E1a, E2a, E3a, E4a])

for i in range(4):
    print("Elastitzitätsmodul",i+1 ,":", Ea[i])

# =============================================================================
#  Auswertung (zweisitig)
# =============================================================================

def xaxis2(L, x):
    return 4*x**3 - 12*L*x**2 + 9*x*L**2 - L**3
    
#def E2
# L = 55.5 cm also gesamtlänge des einspanns

### stab 1 mit plot

x2 = np.array([xaxis2(0.55, pbl), xaxis2(0.55, pbr)])


fit1bl, cov1bl = np.polyfit(x2[0], stab1b[0], 1, cov=True)
poly1d_2 = np.poly1d(fit1bl) 
m1bl = np.array([fit1bl[0], np.sqrt(cov1bl[0,0])])

fit1br, cov1br = np.polyfit(x2[1], stab1b[1], 1, cov=True)
poly1d_2r = np.poly1d(fit1br) 
m1br = np.array([fit1br[0], np.sqrt(cov1br[0,0])])

fig, ax = plt.subplots(ncols= 2, sharey=True, sharex =True)
fig.suptitle("Doppelseitige Einspannung")
ax[0].plot(x2[0], stab1b[0]*10**6, 'x',x2[0], poly1d_2(x2[0])*10**6, 'k--')
ax[1].plot(x2[1], stab1b[1]*10**6, 'x',x2[1], poly1d_2r(x2[1])*10**6, 'k--')
ax[0].title.set_text('linkseitig')
ax[1].title.set_text('rechtsseitig')
ax[0].set_ylabel(r'$\mu m$')
#ax[1].set_ylabel(r'$\mu m$')
ax[0].set_xlabel(r'$(Lx^3 - 12Lx^2 + 9xL^2 - L^3)$')
ax[1].set_xlabel(r'$(Lx^3 - 12Lx^2 + 9xL^2 - L^3)$')

#ax[0].set(adjustable = "box-forced", aspect ="equal")
#ax[1].set(adjustable = "box-forced", aspect ="equal")
plt.tight_layout()


    
# stab 2

fit2bl, cov2bl = np.polyfit(x2[0], stab2b[0], 1, cov=True)
m2bl = np.array([fit2bl[0], np.sqrt(cov2bl[0,0])])   
    
fit2br, cov2br = np.polyfit(x2[1], stab2b[1], 1, cov=True)
m2br = np.array([fit2br[0], np.sqrt(cov2br[0,0])])   
    
# stab 3

fit3bl, cov3bl = np.polyfit(x2[0], stab3b[0], 1, cov=True)
m3bl = np.array([fit2bl[0], np.sqrt(cov3bl[0,0])])   
    
fit3br, cov3br = np.polyfit(x2[1], stab3b[1], 1, cov=True)
m3br = np.array([fit3br[0], np.sqrt(cov3br[0,0])])   
    
# stab 4

fit4bl, cov4bl = np.polyfit(x2[0], stab4b[0], 1, cov=True)
m4bl = np.array([fit4bl[0], np.sqrt(cov4bl[0,0])])   
    
fit4br, cov4br = np.polyfit(x2[1], stab4b[1], 1, cov=True)
m4br = np.array([fit4br[0], np.sqrt(cov4br[0,0])])   
       
    
mbl = np.array([m1bl, m2bl, m3bl, m4bl])   



## Berechnung des Elastizitätsmoduls


def E2(m, mass, I):
    F = 9.81*mass
    dE = abs(F/(48*I*m[0]*m[0]))*m[1]
    return ufloat([round(F/(48*I*m[0]),2), round(dE,2)])
    
    
E1bl = E2(m1bl, stab1_mass[1], I[0])
E2bl = E2(m2bl, stab2_mass[1], I[1])
E3bl = E2(m3bl, stab3_mass[1], I[2])
E4bl = E2(m4bl, stab4_mass[1], I[3])

Ebl = np.array([E1bl, E2bl, E3bl, E4bl])

for i in range(4):
    print("Elastitzitätsmodul_2_links",i+1 ,":", Ebl[i])

E1br = E2(m1br, stab1_mass[1], I[0])
E2br = E2(m2br, stab2_mass[1], I[1])
E3br = E2(m3br, stab3_mass[1], I[2])
E4br = E2(m4br, stab4_mass[1], I[3])

Ebr = np.array([E1br, E2br, E3br, E4br])


for i in range(4):
    print("Elastitzitätsmodul_2_rechts",i+1 ,":", Ebr[i])
    
Eb = np.array([Ebl, Ebr])


# =============================================================================
# =============================================================================
# #  DataFrames
# =============================================================================
# =============================================================================

    
    
data_a = pd.DataFrame({"x": pa,
                     "stab1": stab1a,
                     "stab2": stab2a,
                     "stab3": stab3a,
                     "stab4": stab4a 
                     })
latex(data_a)

data_bl = pd.DataFrame({"x": pbl,
                     "stab1": stab1b[0],
                     "stab2": stab2b[0],
                     "stab3": stab3b[0],
                     "stab4": stab4b[0] 
                     })
latex(data_bl)

data_br = pd.DataFrame({"x": pbr,
                     "stab1": stab1b[1],
                     "stab2": stab2b[1],
                     "stab3": stab3b[1],
                     "stab4": stab4b[1] 
                     })
latex(data_br)


data_Ea = pd.DataFrame({
                     "stab1": np.array([E1a, E1bl, E1br]),
                     "stab2": np.array([E2a, E2bl, E2br]),
                     "stab3": np.array([E3a, E3bl, E3br]),
                     "stab4": np.array([E4a , E4bl, E4br])
                     })
data_Ea.transform(lambda x : x*10**(-7))
latex(data_Ea)







    