#!/usr/bin/env python3

################################################################################
# AUTHOR: João Falcão Neves
# VERSION: 1.000
# DATE: 07/02/2020
#
# Description:
#  Python3 script to filter forced motion forces in interDyMFoam solver
#
# BUGS:
#
# Instructions:
#  To make the script executable go to file directory and type in the terminal:
#    < chmod u+x script_name.py >
#  Otherwise type, each time:
#    < python3 script_name.py >
################################################################################
# Clock
from datetime import datetime
start_time = datetime.now()
################################################################################
# Import modules:
import os
import csv
import pandas as pd
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
################################################################################
# Input Variables ##############################################################
################################################################################
ZZ = 0.5 # Thickness of cell in Z direction

w_prime = 0.5 # w' in [rad/sec²]
g = 9.81 # gravity acceleration [m/sec²]
D = 5 # Draft in [m]

fUnit = 1e-3 # Convert force from N to kN
trInt = [20, 65] # Interval to truncate force, in seconds ---> used to fit curve

#_Raw_forces_file_______________________________________________________________
file = 'forces.dat'

#_File_directory________________________________________________________________
pathToForce = 'postProcessing/forces/0/'

#_Column_Headers________________________________________________________________
fieldnames = ['Time','XPressureF','YPressureF','ZPressureF','XViscousF',
'YViscousF','ZViscousF','XPorousF','YPorousF','ZPorousF','XPressureM',
'YPressureM','ZPressureM','XViscousM','YViscousM','ZViscousM','XPorousM',
'YPorousM','ZPorousM']
################################################################################
# Post Process Forces ##########################################################
################################################################################

freq = np.sqrt(w_prime*g/D) # Frequency used in simulation

forceFiltering = pd.read_csv(pathToForce + file, delim_whitespace=True, skiprows=[0,1,2], header=None, names=fieldnames)

for fieldName in fieldnames:
    forceFiltering[fieldName]= forceFiltering[fieldName].replace(regex=['\(','\)'], value='')

#_Convert_to_float______________________________________________________________
forceFiltering = forceFiltering.astype('float64')

#_Convert_to_kN_and_adimensionalize_force_in_Z__________________________________
forceFiltering.loc[:,['XPressureF','YPressureF','ZPressureF','XViscousF',
'YViscousF','ZViscousF','XPorousF','YPorousF','ZPorousF']] *= (fUnit/ZZ)

#_Write_to_csv_file_____________________________________________________________
forceFiltering.to_csv(pathToForce + 'processedForces.csv', index=False)

truncatedForces = forceFiltering.loc[(forceFiltering['Time'] > trInt[0]) & (forceFiltering['Time'] < trInt[1])]
#_Fit_COS_Function______________________________________________________________
def cosFunc(t, A, phase):
    return A*np.cos((t*freq) + phase)


popt, pcov = curve_fit(cosFunc, truncatedForces.loc[:,'Time'], truncatedForces.loc[:,'YPressureF'])
print(popt, pcov)
plt.plot(truncatedForces.loc[:,'Time'], cosFunc(truncatedForces.loc[:,'Time'], *popt), 'g-', label = 'fit: A=%5.3f kN, phase=%5.3f rad' % tuple(popt))

#_Plots_________________________________________________________________________
'''
x = forceFiltering.loc[:,'Time']

y = forceFiltering.loc[:,'YPressureF']
y2 = forceFiltering.loc[:,'XPressureF']
y3 = forceFiltering.loc[:,'ZPressureF']
y4 = forceFiltering.loc[:,'YViscousF']
y5 = forceFiltering.loc[:,'XViscousF']
y6 = forceFiltering.loc[:,'ZViscousF']
y7 = forceFiltering.loc[:,'YPorousF']
y8 = forceFiltering.loc[:,'XPorousF']
y9 = forceFiltering.loc[:,'ZPorousF']

plt.figure(1)
plt.plot(x, y, label='Y Pressure Force', color='orange')
plt.plot(x, y2, label='X Pressure Force', color='blue')
plt.plot(x, y3, label='Z Pressure Force', color='green')
plt.xlabel('Time [sec]')
plt.ylabel('Force [kN]')
plt.title('Pressure Forces')
plt.legend()

plt.figure(2)
plt.plot(x, y4, label='Y Viscous Force', color='orange')
plt.plot(x, y5, label='X Viscous Force', color='blue')
plt.plot(x, y6, label='Z Viscous Force', color='green')
plt.xlabel('Time [sec]')
plt.ylabel('Force [kN]')
plt.title('Viscous Forces')
plt.legend()

plt.figure(3)
plt.plot(x, y7, label='Y Porous Force', color='orange')
plt.plot(x, y8, label='X Porous Force', color='blue')
plt.plot(x, y9, label='Z Porous Force', color='green')
plt.xlabel('Time [sec]')
plt.ylabel('Force [kN]')
plt.title('Porous Forces')
plt.legend()

plt.figure(4) '''
plt.plot(truncatedForces['Time'], truncatedForces['YPressureF'], label='Truncated Y Pressure Force', color='red')
plt.legend()
plt.show()

################################################################################
print('Finnished!')
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
