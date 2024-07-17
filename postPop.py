#!/usr/bin/python

import os
import sys
import math
import numpy as np
import popFOAM as pop
import matplotlib.pyplot as plt
import scipy.integrate as integrate 

grid = "Fine_InterfaceCompression"
folder_path = "/mnt/Data1/jneves/of10/VerificationAndValidationVal_HeaveBatch/H1.00/RefinementStudy/meshRefinement/wallFunc/H1.00_"+grid+"/"
forces_path = folder_path+"postProcessing/forces/0/"
forces_file = "forces.dat"

if __name__ == "__main__":

    if not os.path.isfile(forces_path+forces_file):
        print("Forces file not found at ", forces_path)
        print("Be sure that the case has been run and you have the right directory!")
        print("Exiting.")
        sys.exit()

    #********************************************************************#
    # User defined variables
    #********************************************************************#   

    g = 9.81 
    rho = 998.2
    draft = 5
    motionAmp = 0.5
    wprime = 1
    w = np.sqrt(wprime*g/draft)
    T = 2.0 * math.pi / w
    Ldeep = g / (2*math.pi) * (T**2) 
    timeStep = T/400
    NT = 5 * T
    ramp = 0#T
    R = draft
    alpha = np.arccos(motionAmp/R)
    Awp = 2 * R * np.sin(alpha)

    #********************************************************************#
    # Create force file and python array
    #********************************************************************#   

    time = np.array([])
    forceX = np.array([])
    forceY = np.array([])    
    time, forceX, forceY, _ = pop.createForceFile(forces_path+forces_file)

    average_forceX = np.mean(forceX)
    average_forceY = np.mean(forceY)
    
    #********************************************************************#
    # Truncate results
    #********************************************************************#   

    truncMin = 2*T 
    truncMax = np.max(time)

    min_truncate_index = np.argmax(time >= truncMin)
    max_truncate_index = np.argmax(time >= truncMax)

    time_truncated = time[min_truncate_index:max_truncate_index]
    forceX_truncated = forceX[min_truncate_index:max_truncate_index]
    forceY_truncated = forceY[min_truncate_index:max_truncate_index]

    #motion_signal = [((time / ramp) if time < ramp else 1) * motionAmp * np.sin(w * time) for time in time_truncated]
    motion_signal = [motionAmp * np.sin(w * time) for time in time_truncated]
    #********************************************************************#
    # Calculate Hydrostatic Force
    #********************************************************************#   

    #hydrostaticforceAVG = np.average(forceY_truncated)
    
    calc = False # set true to calculate hydrostatic force in time, false as the constant initial value

    if calc:
        shape = pop.hullshape(name='cylinder_shape.txt', print=True)
        xvalues = np.array(shape['X'])
        yvalues = np.array(shape['Y'])
        restoringCoeff =  pop.restoringCoefficient(xvalues, yvalues, motion_signal, degree=4, initial_guess=4998)
        hydrostaticforce = restoringCoeff * motion_signal
    else:
        hydrostaticforce = g * rho * math.pi * R**2 / 2 # init hydrostatic force

    #********************************************************************#
    # Correct forces, subtract hydrostatic force -> Add weight = initial buoyancy
    #********************************************************************#   

    forceY_truncated = forceY_truncated - hydrostaticforce

    #********************************************************************#
    # Calculate the force components in and out of phase with the motion 
    #********************************************************************#   

    # Calculate the integral using Simpson's rule
    sync_force = 2 / NT * integrate.simps(forceY_truncated * np.cos(w * time_truncated), time_truncated)
    async_force = 2 / NT * integrate.simps(forceY_truncated * np.sin(w * time_truncated), time_truncated)
    Fa = np.sqrt(sync_force**2+async_force**2)
    phase_shift = np.arcsin(sync_force/Fa) #+ 2 * np.pi

    print(f'\nMotion synch force: {sync_force:.3f} N\nMotion async force: {async_force:.3f} N\nForce amplitude: {Fa:.3f} N\nPhase shift between motion and force: {phase_shift:.3f} rad\nPhase shift between motion and force: {np.degrees(phase_shift):.3f} deg\nPhase shift between motion and force: {phase_shift/w:.3f} seconds\n')

    #********************************************************************#
    # Calculate the hydrodynamic coefficients 
    #********************************************************************#   

    c = 2 * R * rho * g
    a = (c - sync_force / motionAmp) / (w**2) - hydrostaticforce/g
    b = (async_force / (motionAmp * w))
    
    u = a / (rho * (np.pi / 2) * R**2)
    v = b / (rho * (np.pi / 2) * R**2)

    #********************************************************************#
    # Plot and save results 
    #********************************************************************#   

    plt.figure(50)
    
    plt.gca().spines['bottom'].set_position('zero')
    plt.plot(time_truncated, motion_signal/np.max(motion_signal), label='Motion')
    plt.plot(time_truncated, forceY_truncated/np.max(np.abs(forceY_truncated)), label='Force')

    highlight_time = truncMin+T/2 
    
    plt.axvline(x=highlight_time, color='gray', linestyle='--', label='Highlight Time')
    plt.axvline(x=highlight_time + phase_shift/w, color='red', linestyle='--', label='Phase Shift')

    # Annotating the phase difference
    plt.annotate(f'Phase delay: {(phase_shift)/w:.3f} seconds', 
                xy=(highlight_time, 0), xytext=(highlight_time, max(motion_signal)),
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                horizontalalignment='left')

    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Motion and Force Time History with Phase Difference')
    plt.legend()
    plt.savefig(folder_path+"MotionForcePhase.pdf", dpi=300, format='pdf')
    plt.close(50)

    # Open the file in write mode
    with open(folder_path+'results.txt', 'w') as f:
        # Use the write() method instead of print()
        f.write(f"\nSimulation time (s): {12*T:.3f}\n")
        f.write(f'\nWAVE PROPERTIES\nPeriod (s): {T:.3f} \nDeepwater wave length (m): {Ldeep:.3f}\nMeasuring wave at {2*Ldeep} m from buoy'  )
        f.write(f"\nThe force in phase with the motion, F*sin(phi), is: {sync_force:.3f} N")
        f.write(f"\nThe force out of phase with the motion, F*cos(phi), is: {async_force:.3f} N\n")
        f.write(f"\nForce amplitude: {Fa:.3f} N")
        f.write(f'\nPhase shift between motion and force:\n Radians: {phase_shift:.3f}\n Degrees: {np.degrees(phase_shift):.3f}\n')
        f.write(f"\nv': {v:.3f}")
        f.write(f"\nu': {u:.3f}\n")

    #********************************************************************#
    # Calculate Y+ on the floating object 
    #********************************************************************#   

    pop.yplus(folder_path, 'floatingObj')

    #********************************************************************#
    # Calculate the radiated wave height 
    #********************************************************************#   
    '''
    wave = pop.RadiatedWave(distance=Ldeep, waveperiod=T, mainfolderpath=folder_path).freesurfaceelevation()  
    
    print(f'max wave height of {np.max(wave[:,2])} measured at {np.average(wave[:, 1])}m with a measuring position error of {np.max(wave[:,3])}m')
    
    plt.figure(23)
    plt.plot(wave[:,0], wave[:,2], label='free surface elevation')
    plt.xlabel('Time')
    plt.ylabel('Wave amplitude')
    plt.title('Radiated Wave Time History')
    plt.legend()
    plt.savefig(folder_path+"waveHistory.pdf", dpi=300, format='pdf')
    plt.close(23)
    
    np.savetxt(folder_path+'waveElevation.csv', wave, delimiter=',')
    '''