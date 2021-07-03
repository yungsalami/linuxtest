# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 21:44:43 2021

@author: salem
"""
import math as m
#from matplotlib import pyplot as plt
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

#%matplotlib inline

def latex(a):
    print(a.to_latex(decimal = ",", index=False))

# =============================================================================
# =============================================================================
# # statische Methode
# =============================================================================
# =============================================================================


os.chdir(r'C:\Users\salem\OneDrive\Desktop\Physik4\Praktikum4Semester\V204_Wärmeleitung')
data1 = pd.read_csv("GLXportRun1.txt",encoding='utf16',sep='\t').to_numpy()
#print(data1)
t = data1[:,0]/5





# Brass (dick) - T1 und T2


fig1, ax1 = plt.subplots(2)
fig1.suptitle('Messing (dick)')
ax1[0].plot(t,data1[:,1])
ax1[1].plot(t,data1[:,2])
ax1[0].set_xticklabels([])

ax1[1].set_xlabel('Zeit in s')
ax1[1].set_ylabel('Temperatur in C')
ax1[0].set_ylabel('Temperatur in C')


#ax1[2].plot(t,data1[:,2]-data1[:,1])

# Brass (dünn) - T3 und T4



fig2, ax2 = plt.subplots(2)
fig2.suptitle('Messing (dünn)')
ax2[0].plot(t,data1[:,4])
ax2[1].plot(t,data1[:,3])
ax2[0].set_xticklabels([])

ax2[1].set_xlabel('Zeit in s')
ax2[1].set_ylabel('Temperatur in C')
ax2[0].set_ylabel('Temperatur in C')
# Aluminium - T5 und T6

fig3, ax3 = plt.subplots(2)
fig3.suptitle('Aluminium')
ax3[0].plot(t,data1[:,5])
ax3[1].plot(t,data1[:,6])
ax3[0].set_xticklabels([])

ax3[1].set_xlabel('Zeit in s')
ax3[1].set_ylabel('Temperatur in C')
ax3[0].set_ylabel('Temperatur in C')
# stainless steel - T8 und T7

fig4, ax4 = plt.subplots(2)
fig4.suptitle('Edelstahl')
ax4[0].plot(t,data1[:,8])
ax4[1].plot(t,data1[:,7])
ax4[0].set_xticklabels([])
ax4[0].set_ylabel('Temperatur in C')

ax4[1].set_xlabel('Zeit in s')
ax4[1].set_ylabel('Temperatur in C')
#   1: 43.79, 4:37.99, 5:47.98, 8:34.76


#Literaturwerte:
kk = np.array([120.0, 120, 237, 15])
A = np.array([0.48, 0.28 , 0.48, 0.48])*10**(-4)

# beliebige Messzeiten

Messp = [100, 150, 200, 330, 550]


dT = np.empty([5,4])
Mat = np.array([ [2,1], [3,4], [6,5], [7,8]])
for i in range(4):#  i über Material
    for k in range(5):
        print(k)
        print(abs(data1[Messp[k],Mat[i,0]] - data1[Messp[k],Mat[i,1]]))
        dT[k,i] = abs(data1[Messp[k],Mat[i,0]] - data1[Messp[k],Mat[i,1]])
        
        print(dT[k,i])
    
dTl = pd.DataFrame({"Messzeitpunkt": Messp,
                    "Messing(dick)": dT[:,0],
                    "Messing(dünn)": dT[:,1],
                    "Alumium": dT[:,2],
                    "Edelstahl": dT[:,3]
                    })


    
def warmestrom(k, A, dT):
    return k*A*dT

DQ = np.zeros([5,4])
    
for i in range(4):
    for k in range(5):#
        print(k)
        ddT = dT[k,i]/0.03      # durch dx teilen !
        print(ddT)
        DQ[k,i] = -kk[i]*A[i]*ddT
    
print(dT)
print(DQ)
dQ = DQ

dQl = pd.DataFrame({"Messzeitpunkt": Messp,
                    "Messing(dick)": dQ[:,0],
                    "Messing(dünn)": dQ[:,1],
                    "Alumium": dQ[:,2],
                    "Edelstahl": dQ[:,3]
                    })


qbrasdick = np.array([[np.mean(dQ[:,0]), np.std(dQ[:,0])],
                      [np.mean(dQ[:,1]), np.std(dQ[:,1])],
                      [np.mean(dQ[:,2]), np.std(dQ[:,2])],
                      [np.mean(dQ[:,3]), np.std(dQ[:,3])],
                      ])

# stat = pd.DataFrame({"brass dick": dT[:,1],
#                      "brass thin": dT[:,2]},
#                     "alu": dT)


stat1 = pd.DataFrame(dT, columns=("brass dick", "brass thin", "alu", "edelstahl"))



stat2 = pd.DataFrame(DQ, columns=("brass dick", "brass thin", "alu", "edelstahl"))

latex(stat1)
latex(stat2)

############## 


fig5, ax5 = plt.subplots(2)
ax5[0].plot(t, data1[:,7]-data1[:,8], 'r-')
ax5[1].plot(t, data1[:,2]-data1[:,1])
ax5[0].title.set_text(" 'T7-T8'-Verlauf ")
ax5[1].title.set_text(" 'T2-T1'-Verlauf ")
ax5[1].set_xlabel('Zeit in s')
ax5[1].set_ylabel('Temperatur in C')
ax5[0].set_ylabel('Temperatur in C')

plt.tight_layout()














# =============================================================================
# =============================================================================
# # dynamische Methode
# =============================================================================
# =============================================================================

# =============================================================================
# ## temperaturverlauf des Breiten messingstabes T1 und T2 ###########################
# =============================================================================

os.chdir(r'C:\Users\salem\OneDrive\Desktop\Physik4\Praktikum4Semester\V204_Wärmeleitung')
data2 = pd.read_csv("GLXportRun1_b.txt",encoding='utf16',sep='\t').to_numpy()
t2 = data2[:,0]/2


## Extrema von T2
peaks2 = find_peaks(data2[:,2])[0]
dip2 = find_peaks(-data2[:,2])[0]
ext2 = np.concatenate((peaks2,dip2),axis=None)
null = np.array([0])
ext2 = np.concatenate((ext2,null), axis=None)
ext2.sort()
## Extrema von T1
peaks1 = find_peaks(data2[:,1])[0]
dip1 = find_peaks(-data2[:,1])[0]
ext1 = np.concatenate((peaks1,dip1),axis=None)
ext1 = np.concatenate((ext1,null), axis=None)
ext1.sort()



fig7, ax7 = plt.subplots(2)

ax7[0].plot(t2,data2[:,1])
ax7[1].plot(t2,data2[:,2])
ax7[0].title.set_text("T1")
ax7[1].title.set_text("T2")
ax7[1].set_xlabel('Zeit in s')
ax7[1].set_ylabel('Temperatur in C')
ax7[0].set_ylabel('Temperatur in C')

plt.tight_layout()

plt.figure()
plt.plot(t2[ext2],data2[ext2,2],'.r')
plt.plot(t2, data2[:,2])
plt.xlabel('Zeit in s')
plt.ylabel('Temperatur in C')
plt.ylabel('Temperatur in C')


plt.grid()

## Amplituden bestimmen:
Amp2 = np.empty(11)
for i in np.arange(0,11):
    print(i)
    Amp2[i] = data2[ext2[(i)*2+1],2] - data2[ext2[i*2],2]

print(Amp2)

plt.figure()
plt.plot(t2[ext1],data2[ext1,1],'.r')
plt.plot(t2, data2[:,1])
plt.grid()
plt.xlabel('Zeit in s')
plt.ylabel('Temperatur in C')
plt.ylabel('Temperatur in C')

## ablesen auf Textfile amk




Amp1 = np.array([6.04, 4.62, 3.85, 1.35, 3.1, 2.89, 2.73, 2.53, 2.46, 2.37, 2.34  ])

print(Amp1)
opop = data2[ext1[1*2+1],1] - data2[ext1[1*2],1]



### Phasendifferenz

Amp2max = t2[ext2[1:-1:2]]/2
Amp1max = t2[[156, 298, 448, 565, 683, 842, 1000, 1160, 1321, 1477, 1639]]/2

phase_bras = Amp2max - Amp1max
print(phase_bras)


AmplitudenBras = pd.DataFrame({"T1-Amplituden": Amp1,
                           "T2-Amplituden": Amp2,
                           "Phase": phase_bras
                           })
latex(AmplitudenBras)













# =============================================================================
# ### Temperaturverlauf Alu T5 und T6 #####################################
# =============================================================================



## Extrema von T5
peaks5 = find_peaks(data2[:,5])[0]
dip5 = find_peaks(-data2[:,5])[0]
ext5 = np.concatenate((peaks5,dip5),axis=None)
null = np.array([0])
ext5 = np.concatenate((ext5,null), axis=None)
ext5.sort()
ext5 = np.delete(ext5, 1)
## Extrema von T6
peaks6 = find_peaks(data2[:,6])[0]
dip6 = find_peaks(-data2[:,6])[0]
ext6 = np.concatenate((peaks6,dip6),axis=None)
ext6 = np.concatenate((ext6,null), axis=None)
ext6.sort()

zt = t2[ext5]
zt2 = t2[ext6]
plt.figure()
plt.plot(t2, data2[:,5],'b-', label = "T5")
plt.plot(t2, data2[:,6], 'r-', label = "T6")
plt.legend()

plt.xlabel('Zeit in s')
plt.ylabel('Temperatur in C')
plt.ylabel('Temperatur in C')
plt.title("Aluminium (T5)")



Amp5 = np.empty(11)
Amp6 = np.empty(11)
for i in np.arange(0,11):
    print(i)
    Amp5[i] = data2[ext5[(i)*2+1],5] - data2[ext5[i*2],5]
    Amp6[i] = data2[ext6[(i)*2+1],6] - data2[ext6[i*2],6]

print(Amp5)
print(Amp6)

## zeitindex von Amplituden

AmpIndex5 = t2[ext5[1:22:2]]
AmpIndex6 = t2[ext6[1:-1:2]]
phase_alu = AmpIndex6 - AmpIndex5




AmplitudenAlu = pd.DataFrame({"T5-Amplituden": Amp5,
                           "T6-Amplituden": Amp6,
                           "Phase": phase_alu
                           })
latex(AmplitudenAlu)




# =============================================================================
# ###### Temperaturverlauf Edelstahl T7 und T8 ####################
# =============================================================================
ext8 = np.array([0, 395, 477, 765, 890, 1170, 1295, 1560, 1698, 1953, 2107, 2353])

os.chdir(r'C:\Users\salem\OneDrive\Desktop\Physik4\Praktikum4Semester\V204_Wärmeleitung')
data3 = pd.read_csv("GLXportRun1c.txt",encoding='utf16',sep='\t').to_numpy()
t3 = data3[:,0]






## Extrema von T5
peaks7 = find_peaks(data3[:,7])[0]
dip7 = find_peaks(-data3[:,7])[0]
ext7 = np.concatenate((peaks7,dip7),axis=None)
null = np.array([0])
ext7 = np.concatenate((ext7,null), axis=None)
ext7.sort()

## Extrema von T6
# peaks8 = find_peaks(data3[:,8])[0]
# dip8 = find_peaks(-data3[:,8])[0]
# ext8 = np.concatenate((peaks8,dip8),axis=None)
# ext8 = np.concatenate((ext8,null), axis=None)
# ext8.sort()

zt = t2[ext5]
zt2 = t2[ext6]



plt.figure()
plt.plot(t3, data3[:,7], label = "T7" )
plt.plot(t3, data3[:,8], label = "T8"  )

plt.legend()

plt.xlabel('Zeit in s')
plt.ylabel('Temperatur in C')

plt.plot(t3[ext7], data3[ext7,7],'b.')
plt.plot(t3[ext8], data3[ext8,8], 'r.')

Amp7 = np.empty(6)

for i in np.arange(0,6):
    print(i)
    Amp7[i] = data3[ext7[(i)*2+1],7] - data3[ext5[i*2],7]
    print(Amp7[i])

## T8 Amplituden müssen abgelesen

Amp8 = np.array([3.94, 3.1, 2.91, 2.62, 2.32, 2.19 ])


print(Amp7)
print(Amp8)

## zeitindex von Amplituden



AmpIndex7 = t3[ext7[1:22:2]]/2
AmpIndex8 = t3[[395, 765, 1170, 1560, 1953, 2353]]/2
phase_stahl = AmpIndex8 - AmpIndex7




AmplitudenStahl = pd.DataFrame({"T7-Amplituden": Amp7,
                           "T8-Amplituden": Amp8,
                           "Phase": phase_stahl
                           })
latex(AmplitudenStahl)





def wlf(p, c, a1, a2, dt):
    return p*c*(0.03*0.03)/(2*dt*np.log(a1/a2))



kbras= wlf(8520, 385, Amp2, Amp1, -phase_bras)  
kalu= wlf(2800, 830, Amp6, Amp5, -phase_alu)  
kstahl= wlf(8000, 400, Amp7, Amp8, phase_stahl) 


# =============================================================================
# Phasengeschwindigkeit
# =============================================================================

def phv(p, c, k, T):
    w = 2*np.pi/T
    return np.sqrt(2*k*w/(p*c))

vbras = phv(8520, 385, kbras, 80)
valu = phv(2800, 830, kalu, 80)
vstahl = phv(8000, 400, kstahl, 200)

AmplitudenStahl = pd.DataFrame({"T7-Amplituden": Amp7,
                           "T8-Amplituden": Amp8,
                           "Phase": phase_stahl,
                           "warmeleitfaehigkeit": kstahl,
                           "vphase": vstahl
                           })
latex(AmplitudenStahl)

AmplitudenAlu = pd.DataFrame({"T5-Amplituden": Amp5,
                           "T6-Amplituden": Amp6,
                           "Phase": -phase_alu,
                           "warmeleitfaehigkeit": kalu,
                           "vphase": valu
                           })
latex(AmplitudenAlu)

AmplitudenBras = pd.DataFrame({"T1-Amplituden": Amp1,
                           "T2-Amplituden": Amp2,
                           "Phase": -phase_bras,
                           "warmeleitfaehigkeit": kbras,
                           "vphase": vbras
                           })
latex(AmplitudenBras)
mean = np.array([[np.mean(vstahl), np.std(vstahl)],[np.mean(valu), np.std(valu)],[np.mean(vbras), np.std(vbras)]])
mean = mean*10**5 # also 10^-5 m/s
print(mean)

### theoretischer Phasengeschwindigkeit

vbras = phv(8520, 385, 109, 80)
valu = phv(2800, 830, 205, 80)
vstahl = phv(8000, 400, 16, 200)


mean = np.array([[np.mean(vstahl), np.std(vstahl)],[np.mean(valu), np.std(valu)],[np.mean(vbras), np.std(vbras)]])
mean = mean*10**5 # also 10^-5 m/s
print(mean)


