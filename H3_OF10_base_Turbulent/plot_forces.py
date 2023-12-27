#!/usr/bin/python

import os
import sys
import math
import numpy as np
from scipy.signal import argrelextrema



class HydroCoeff:

    def __init__(self, phaselag, hydrodynamicforce, motionamplitude, w):
        # Passing arguments to instance attributes
        self.phaselag = phaselag
        self.hydrodynamicforce = hydrodynamicforce
        self.motionamplitude = motionamplitude
        self.w = w
        # Calculates damping and assigns as instance attribute:
        self.damping = - self.hydrodynamicforce * np.sin(self.phaselag) / (self.motionamplitude * self.w)
        # Calculates damping and assigns as instance attribute:
        self.addedmass = self.hydrodynamicforce * np.cos(self.phaselag) / (self.motionamplitude * self.w ** 2)



forces_file = "postProcessing/forces/0/forces.dat"

if not os.path.isfile(forces_file):
	print("Forces file not found at ", forces_file)
	print("Be sure that the case has been run and you have the right directory!")
	print("Exiting.")
	sys.exit()

def line2dict(line):
	tokens_unprocessed = line.split()
	tokens = [x.replace(")","").replace("(","") for x in tokens_unprocessed]
	floats = [float(x) for x in tokens]
	data_dict = {}
	data_dict['Time'] = floats[0]
	force_dict = {}
	force_dict['pressure'] = floats[1:4]
	force_dict['viscous'] = floats[4:7]
	force_dict['porous'] = floats[7:10]
	moment_dict = {}
	moment_dict['pressure'] = floats[10:13]
	moment_dict['viscous'] = floats[13:16]
	moment_dict['porous'] = floats[16:19]
	data_dict['forces'] = force_dict
	data_dict['moments'] = moment_dict
	return data_dict

time = []
forceY = []
#lift = []
#moment = []
with open(forces_file,"r") as datafile:
	for line in datafile:
		if line[0] == "#":
			continue
		data_dict = line2dict(line)
		time += [data_dict['Time']]
		forceY += [data_dict['forces']['pressure'][1] + data_dict['forces']['viscous'][1]]
		#lift += [data_dict['forces']['pressure'][1] + data_dict['forces']['viscous'][1]]
		#moment += [data_dict['moments']['pressure'][2] + data_dict['moments']['viscous'][2]]
datafile.close()

outputfile = open('forces.txt','w')
for i in range(0,len(time)):
	outputfile.write(str(time[i])+' '+str(forceY[i])+'\n') #+str(drag[i])+' '+str(moment[i])+'\n')
outputfile.close()

os.system("./gnuplot_script.sh")


trunc_time = []
trunc_forceY = []
Damping = []
AddedMass = []

draft = 5
motionAmp = 0.5
wprime = 0.2
w = np.sqrt(wprime*9.81/draft)
truncMax = 65
truncMin = 30

with open("forces.txt", 'r') as file:
    for line in file:
        t, f = map(float, line.strip().split())
        if truncMin < t < truncMax:
            trunc_time.append(t)
            trunc_forceY.append(f)

trunc_time = np.array(trunc_time)  # Convert list to numpy array for operations
motion_signal = motionAmp * np.sin(w * trunc_time)

# Calculate the cross-correlation
cross_correlation = np.correlate(trunc_forceY - np.mean(trunc_forceY), motion_signal - np.mean(motion_signal), 'full')

# Calculate the delays for each cross-correlation value
delays = np.arange(1-len(motion_signal), len(motion_signal))

# Find the delay corresponding to the maximum cross-correlation value
delay = delays[np.argmax(cross_correlation)]

print(f"The phase lag (delay) between the force and motion is: {delay} time units")

						
# Identify the indices of the peaks in the truncated force data
peak_indices = argrelextrema(np.array(trunc_forceY), np.greater)[0]
peak_forces = [trunc_forceY[i] for i in peak_indices]

for peak_force in peak_forces:
    # Instantiate the HydroCoeff class object
    hydro_coeff = HydroCoeff(delay, peak_force, motionAmp, w)
    damping = hydro_coeff.damping/(np.pi() / 2 * 998.2 * draft**2)
    added_mass = hydro_coeff.addedmass/(np.pi() / 2 * 998.2 * draft**2)

    # Print the calculated damping for each peak
    print(f"For peak force {peak_force}, the calculated damping is: {damping} units")			

