#!/usr/bin/python3
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
#    < chmod u+x script name.py >
#  Otherwise type, each time:
#    < python3 script name.py >


################################################################################
# Import modules:
import tkinter
from tkinter import *
from tkinter import filedialog
from datetime import datetime
import hydrocoeffs as hc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import math
from sklearn.metrics import r2_score

#################################################################################
g = 9.81  # Gravity acceleration
rho = 1000  # Water density kg/m³
radius = 5  # Cylinder radius in m


def cos_func(t, amplitude, phase):
    return amplitude * np.cos((t * freq) + phase)


def nondim(coefficient, radius):  # only works for cylinder shapes (radius)
    return coefficient / (rho * (math.pi / 2) * radius ** 2)


def my_range(start, end, step):  # Module to help for loop
    while start <= end:
        yield start
        start += step


def save():  # Module to get GUI input parameters
    aa = E1.get()
    bb = E2.get()
    cc = E3.get()
    dd = E4.get()
    ee = E5.get()
    ff = E6.get()
    gg = E7.get()
    hh = E8.get()
    ii = E9.get()
    main.destroy()

    global params
    params = [aa, bb, cc, dd, ee, ff, gg, hh, ii]


# Start of GUI
main = tkinter.Tk()

main.title("OpenFOAM Post-Processing app")

# Ask for directory of the OpenFOAM case
pathToMainFolder = tkinter.filedialog.askdirectory(initialdir='~/OpenFOAM/OpenFOAM-2.4.0/')
# pathToMainFolder = main.directory
print("\nYou are working within the following directory:\n" + pathToMainFolder + "\n")

# Columns headers
L0 = Label(main, text="Parameter").grid(column=0, row=1, sticky=N)
L1 = Label(main, text="Unit").grid(column=3, row=1, sticky=N)
L2 = Label(main, text="Input").grid(columnspan=2, column=1, row=1, sticky=N)

# First line of GUI
L3 = Label(main, text="Z thickness").grid(column=0, row=3, sticky=N)
L4 = Label(main, text="[m]:").grid(column=3, row=3, sticky=W)
E1 = Entry(main)
E1.grid(columnspan=2, column=1, row=3, sticky=W + E + N + S)
E1.insert(END, "0.5")

# Second line of GUI
L5 = Label(main, text="w").grid(column=0, row=4, sticky=N)
L6 = Label(main, text="[-]: ").grid(column=3, row=4, sticky=W)
E2 = Entry(main)
E2.grid(columnspan=2, column=1, row=4, sticky=W + E + N + S)
E2.insert(END, "0.5")

# Third line of GUI
L9 = Label(main, text="Draft").grid(column=0, row=5, sticky=N)
L10 = Label(main, text="[m]").grid(column=3, row=5, sticky=W)
E3 = Entry(main)
E3.grid(columnspan=2, column=1, row=5, sticky=W + E + N + S)
E3.insert(END, "5")

# Fourth line of GUI
L11 = Label(main, text="Conversion factor, ex: 1e-3 [N --> kN]: ").grid(column=0, row=6, sticky=N)
L12 = Label(main, text="[-]").grid(column=3, row=6, sticky=W)
E4 = Entry(main)
E4.grid(columnspan=2, column=1, row=6, sticky=W + E + N + S)
E4.insert(END, "1e-3")

# Fifth line of GUI
L13 = Label(main, text="Interval to truncate [min][max] ").grid(column=0, row=7, sticky=N)
L14 = Label(main, text="[s]").grid(column=3, row=7, sticky=W)
E5 = Entry(main)
E5.grid(column=1, row=7, sticky=W + E + N + S)
E5.insert(END, "20")
E6 = Entry(main)
E6.grid(column=2, row=7, sticky=W + E + N + S)
E6.insert(END, "70")

# Sixth line of GUI
L15 = Label(main, text="Transient time ").grid(column=0, row=8, sticky=N)
L16 = Label(main, text="[s]").grid(column=3, row=8, sticky=W)
E7 = Entry(main)
E7.grid(columnspan=2, column=1, row=8, sticky=W + E + N + S)
E7.insert(END, "12.5")

# Seventh line of GUI
L17 = Label(main, text="Motion amplitude (to keep linearity use max 10% of draft) ").grid(column=0, row=9, sticky=N)
L18 = Label(main, text="[m]").grid(column=3, row=9, sticky=W)
E8 = Entry(main)
E8.grid(columnspan=2, column=1, row=9, sticky=W + E + N + S)
E8.insert(END, "0.5")

# Eight line of GUI
L19 = Label(main).grid(column=0, row=10, sticky=N)
L20 = Label(main, text="[s]").grid(column=3, row=10, sticky=W)
E9 = Entry(main)
E9.grid(columnspan=2, column=1, row=10, sticky=W + E + N + S)
E9.insert(END, "75")

# Nine line of GUI
Lspace = Label(main).grid(columnspan=4, column=0, row=1, sticky=N)

# Run button
runButton = Button(main, text="Run", command=save).grid(column=4, row=11, sticky=N)

mainloop()  # End of GUI

# Start clock
starttime = datetime.now()

# Get parameters from GUI into program script
zz = float(params[0])
wprime = float(params[1])
d = float(params[2])
funit = float(params[3])
trintmin = float(params[4])
trintmax = float(params[5])
transienttime = float(params[6])
motionamplitude = float(params[7])
runtime = float(params[8])
trInt = [trintmin, trintmax]

print("User input:",
      "\n Z thickness: {0} m".format(str(zz)),
      "\n w`: {0}".format(str(wprime)),
      "\n Floating object draft: {0} m".format(str(d)),
      "\n Force conversion factor: {0}".format(str(funit)),
      "\n Truncate time interval: {0} to {1} sec".format(str(trintmin), str(trintmax)),
      "\n Transient evolution time: {0} sec".format(str(transienttime)),
      "\n Motion amplitude: {0} m".format(str(motionamplitude)),
      "\n Simulation runtime: {0} sec\n".format(str(runtime)))

freq = np.sqrt(wprime * g / d)  # Frequency used in simulation

# Create dataFrame for motion
motionData = pd.DataFrame(columns=["time", "vertPosition", "vertVelocity", "vertAcceleration"])

for t in my_range(0, runtime, 0.01):
    if t <= transienttime:
        vertposition = motionamplitude * (-np.sin(freq * t)) * t / transienttime
        vertvelocity = freq * motionamplitude * (-np.cos(freq * t)) * t / transienttime
        vertacceleration = freq * freq * motionamplitude * (np.sin(freq * t)) * t / transienttime

        motionData = motionData.append({"time": t, "vertPosition": vertposition, "vertVelocity": vertvelocity,
                                        "vertAcceleration": vertacceleration}, ignore_index=True)
    else:
        vertposition = motionamplitude * (-np.sin(freq * t))
        vertvelocity = freq * motionamplitude * (-np.cos(freq * t))
        vertacceleration = freq * freq * motionamplitude * (np.sin(freq * t))

        motionData = motionData.append({"time": t, "vertPosition": vertposition, "vertVelocity": vertvelocity,
                                        "vertAcceleration": vertacceleration}, ignore_index=True)

# Create force dataFrame from OpenFOAM output files
#  Raw forces file
file = "forces.dat"

#  File directory
pathToForce = pathToMainFolder + "/postProcessing/forces/0/"

#  Column Headers
fieldnames = ["Time", "XPressureF", "YPressureF", "ZPressureF", "XViscousF",
              "YViscousF", "ZViscousF", "XPorousF", "YPorousF", "ZPorousF", "XPressureM",
              "YPressureM", "ZPressureM", "XViscousM", "YViscousM", "ZViscousM", "XPorousM",
              "YPorousM", "ZPorousM"]

# Open forces.dat as dataFrame with above the headers
forceFiltering = pd.read_csv(pathToForce + file, delim_whitespace=True, skiprows=[0, 1, 2],
                             header=None, names=fieldnames)

# Filter "/", "\", "(" and ")" from forces.dat
forceFiltering = forceFiltering.replace(regex=["\(", "\)"], value="")

# Convert to float
forceFiltering = forceFiltering.astype("float64")

# Convert to kN and adimensionalize force in Z
forceFiltering.loc[:, ["XPressureF", "YPressureF", "ZPressureF", "XViscousF", "YViscousF",
                       "ZViscousF", "XPorousF", "YPorousF", "ZPorousF"]] *= (funit / zz)

# Write cleaned forces to csv file "processedForces.csv" located in same directory
forceFiltering.to_csv(pathToForce + "processedForces.csv", index=False)

# Truncate force and motions
truncatedForces = forceFiltering.loc[(forceFiltering["Time"] > trInt[0]) & (forceFiltering["Time"] < trInt[1])]
truncatedMotionData = motionData.loc[(motionData["time"] > trInt[0]) & (motionData["time"] < trInt[1])]

# Fit cos function to force data
popt, pcov = curve_fit(cos_func, truncatedForces.loc[:, 'Time'], truncatedForces.loc[:, 'YPressureF'])
forceAmplitude = popt[0]  # Maximum force amplitude given by cos curve fit
forcePhase = popt[1]  # Force phase given by cos curve fit

ypv = truncatedForces['YPressureF'].values  # Y pressure force values to use in R² calculation
n = len(truncatedForces.loc[:, 'Time'])  # Length of dataframe to use to calculate values for predicted Y force
predictedYPressureF = truncatedForces['Time'].apply(lambda x: forceAmplitude * np.cos(x * freq + forcePhase))

rr = r2_score(predictedYPressureF, ypv)  # R² value for force fit

plt.figure(1)
plt.plot(truncatedForces['Time'], truncatedForces['YPressureF'], label='Truncated Y Pressure Force', color='red')
plt.plot(truncatedForces.loc[:, 'Time'], cos_func(truncatedForces.loc[:, 'Time'], *popt), 'g-',
         label='Fit: force={:.3f}kN, phase={:.3f}rad\nR²: {:.4f}'.format(forceAmplitude, forcePhase, rr))
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

popt, pcov = curve_fit(cos_func, truncatedMotionData.loc[:, 'time'], truncatedMotionData.loc[:, 'vertPosition'])
motionAmplitude = popt[0]
motionPhase = popt[1]

mDataV = truncatedMotionData['vertPosition'].values  # Object vertical position values to use in R² calculation
n1 = len(truncatedMotionData.loc[:, 'time'])  # Length of dataframe to use to calculate values for predicted motion data
predictedMData = truncatedMotionData['time'].apply(lambda x: motionAmplitude * np.cos(x * freq + motionPhase))

rr1 = r2_score(predictedMData, mDataV)  # R² value for force fit
'''
plt.figure(2)
plt.plot(truncatedMotionData['time'], truncatedMotionData['vertPosition'], label='Truncated Y Motion', color='red')
plt.plot(truncatedMotionData.loc[:, 'time'], cos_func(truncatedMotionData.loc[:, 'time'], *popt), 'g-',
         label='Fit: Motion Amplitude={:.3f}m, Motion phase={:.3f}rad\nR²: {:.4f}'.format(motionAmplitude,
          motionPhase, rr1))
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
plt.show()
'''
# Phase lag
phaseLag = forcePhase - motionPhase

# Wave sampling
csvfile = tkinter.filedialog.askopenfilename(initialdir=pathToMainFolder, title="Select wave elevation file")
csvwavefile = pd.read_csv(csvfile)
# csvwavefile = csvwavefile.sort_values(by='Time')

plt.figure(3)
plt.plot(csvwavefile['Time'], csvwavefile['Y'], label='Truncated wave elevation', color='red')
plt.xlabel('Time [sec]')
plt.show(block=FALSE)


def gui2():
    jj = E10.get()
    kk = E11.get()
    main.destroy()
    global params
    params = [jj, kk]


plt.show(block=False)

main = tkinter.Tk()  # Start of GUI1
main.title("Surface elevation data processing")

Label(main, text="Interval to truncate wave surface[min][max] ").grid(column=0, row=7, sticky=N)
Label(main, text="[s]").grid(column=3, row=7, sticky=W)
E10 = Entry(main)
E10.grid(column=1, row=7, sticky=W + E + N + S)
E11 = Entry(main)
E11.grid(column=2, row=7, sticky=W + E + N + S)
runButton = Button(main, text="Run", command=gui2).grid(column=4, row=11, sticky=N)  # Run button

mainloop()  # End of GUI

minTime = float(params[0])
maxTime = float(params[1])

truncatedSortedWave = csvwavefile.loc[(csvwavefile["Time"] >= minTime) & (csvwavefile["Time"] <= maxTime)]

popt, pcov = curve_fit(cos_func, truncatedSortedWave.loc[:, 'Time'], truncatedSortedWave.loc[:, 'Y'])
waveAmplitude = popt[0]
wavePhase = popt[1]

plt.plot(truncatedSortedWave['Time'], truncatedSortedWave['Y'], label='Truncated surface elevation', color='red')
plt.plot(truncatedSortedWave.loc[:, 'Time'], cos_func(truncatedSortedWave.loc[:, 'Time'], *popt), 'g-',
         label='fit: A=%5.3f m, phase=%5.3f rad' % tuple(popt))
plt.legend()
plt.show()

# Hydrodynamic coeffs using Uzunoglu method:
Ucoeffs = hc.UzunogluMethod(phaseLag, forceAmplitude / funit, motionAmplitude, freq)

nonDimUDamping = nondim(Ucoeffs.damping, radius)
nonDimUAddedMass = nondim(Ucoeffs.addedmass, radius)

print(Ucoeffs.damping, Ucoeffs.addedmass, nonDimUDamping, nonDimUAddedMass)
'''
# Hydrodynamic coeffs using Jorge's method:
Jcoeffs = hc.JorgeMethod(motion)
Jcoeffs.damping(rho, freq, waveAmplitude, motionAmplitude)

nonDimJDamping = nondim(Jcoeffs.damping, radius)

print("\nCalculated values:",
      "\n Oscillation frequency (w): {0} rad/s".format(str(freq)),
      "\n Force Amplitude: {0} kN".format(str(forceAmplitude)),
      "\n Motion Amplitude: {0} m".format(str(motionAmplitude)),
      "\n Phase lag between motion and force is: {0} rad".format(str(phaseLag)),
      "\n Wave Amplitude: {0} m".format(str(waveAmplitude)),
      "\n\n Uzunoglu's damping: {0} N.s/m ".format(str(Ucoeffs.damping)),
      "\n Jorge's damping: {0} N.s/m".format(str(Jcoeffs.damping)),
      "\n\n Uzunoglu's added mass is: {0} kg".format(str(Ucoeffs.addedmass)),
      "\n\n Uzunoglu's non dim damping: {0}".format(str(nonDimUDamping)),
      "\n Jorge's non dim damping: {0}".format(str(nonDimJDamping)),
      "\n\n Uzunoglu's non dim added mass is: {0}".format(str(nonDimUAddedMass)))
'''
