#!/usr/bin/python

import os
import sys
import math
import popTools as pop
import numpy as np
from scipy import signal 
from scipy import integrate
from pyfiglet import Figlet
 

grid = "H1.00_predictorON_t100_L0.5"#/RefinementStudy/meshRefinement/wallFunc/interfaceCompression/H1.00_Coarsest_InterfaceCompression"
folder_path = "/media/joaofn/nvme-WD/"+grid+"/"
forces_path = folder_path+"postProcessing/forces/0/"
forces_file = "forces.dat"

if __name__ == "__main__":
    
    text = Figlet(font='slant')
    print(text.renderText("Let's popFOAM"))

    if not os.path.isfile(forces_path+forces_file):
        print("Forces file not found at ", forces_path)
        print("Be sure that the case has been run and you have the right directory!")
        print("Exiting.")
        sys.exit()
    else:
        print("\n\ncase: " + grid +"\n\n")
    ##################################################################################################################
    # Initialization
    ##################################################################################################################

    # Var init
    time = np.array([])
    forceX = np.array([])
    forceY = np.array([])

    time, forceX, forceY, _ = pop.createForceFile(forces_path+forces_file)
    
    g = 9.81                            # acceleration of gravity
    rho = 998.2                         # water density -> be sure to put same as in the simulation!
    draft = 5                           # draft
    motionAmp = 0.5                     # motion amplitude
    wprime = 1                          # normalized radial frequency
    w = np.sqrt(wprime * g / draft)     # radial frequency    
    velAmp = motionAmp * w              # velocity amplitude
    accelAmp = velAmp * w               # acceleration amplitude
    T = 2.0 * math.pi / w               # period
    freq = 1 / T                        # frequency    
    Ldeep = g / (2 * math.pi) * (T**2)  # length of wave in deep water
    truncMax = np.max(time)             # max time to analyze
    truncMin = 1.9 * T                  # min time to analyze
    ramp = 0 * T                        # transient time
    R = draft                           # radius of the cylinder = draft
    Z = 1                               # 2D domain depth in Z    
    Awp = 2 * R * Z                     # waterplane area
    restoringCoeff = Awp * rho * g      # restoring coefficient
    twoL = 2 * Ldeep                    # 2 * length of deep water wave 

    try:
        # Check for timestep consistency
        inconsistent_steps = pop.check_time_step_consistency(time, tolerance=1e-6)

        if not inconsistent_steps:
            print("Time step is constant")
        else:
            print("Time step is not constant at the following indices and values:")
            for idx, val in inconsistent_steps:
                print(f"Index {idx}: Time difference {val}")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Full motions calculation
    full_motion_signal = [((t / ramp) if t < ramp else 1) * motionAmp * np.sin(w * t) for t in time]
    full_velocity_signal = [((t / ramp) if t < ramp else 1) * w * motionAmp * np.cos(w * t) for t in time]
    full_acceleration_signal = [-((t / ramp) if t < ramp else 1) * w**2 * motionAmp * np.sin(w * t) for t in time]

    pop.makeplot(title='Motion',
                 x=time,
                 y=[full_motion_signal, full_velocity_signal, full_acceleration_signal], 
                 xlabel='Time (s)',
                 ylabel='Amplitude',
                 label=['motion (m)', 'velocity (m/s)', 'acceleration (m/s^2)'], 
                 folder_path=folder_path, 
                 figurename='motion')


    ##################################################################################################################
    # Time, motion and force truncation
    ##################################################################################################################

    # Time truncation    
    min_truncate_index = np.argmax(time >= truncMin)
    max_truncate_index = np.argmax(time >= truncMax)

    time_truncated = time[min_truncate_index:max_truncate_index]

    motion_signal = [((t / ramp) if t < ramp else 1) * motionAmp * np.sin(w * t) for t in time_truncated]
    velocity_signal = [((t / ramp) if t < ramp else 1) * w * motionAmp * np.cos(w * t) for t in time_truncated]
    acceleration_signal = [-((t / ramp) if t < ramp else 1) * w**2 * motionAmp * np.sin(w * t) for t in time_truncated]    
    
    # Force truncation
    forceX_truncated = np.array(forceX[min_truncate_index:max_truncate_index])
    forceY_truncated = np.array(forceY[min_truncate_index:max_truncate_index]) #- restoringCoeff * np.array(motion_signal)

    # Plot motions
    pop.makeplot('Truncated Motion',
                  time_truncated, [motion_signal, velocity_signal, acceleration_signal], 
                 'Time (s)', 'Amplitude',
                 ['motion (m)', 'velocity (m/s)', 'acceleration (m/s^2)'], 
                 folder_path, 'truncated_motion')


    ##################################################################################################################
    # Freesurface calculation
    ##################################################################################################################

    pop.yplus(folder_path, 'floatingObj')
    location = 1
    radiated_wave = pop.RadiatedWave(waveperiod=T, mainfolderpath=folder_path)
    radiated_wave.freesurfaceelevation(probe=location, relBottom=False)
    wave_history = radiated_wave.wave_history
    
    # Time to truncate wave != truncate min
    minTime = truncMin
    mask = wave_history[0] > minTime
    wave_filtered = wave_history[:, mask]

    # Find wave peaks
    pos_wavepeaks_indices, _ = signal.find_peaks(wave_filtered[1], threshold=None)
    neg_wavepeaks_indices, _ = signal.find_peaks(-1*wave_filtered[1], threshold=None)
    
    average_positive_amplitude = np.average(wave_filtered[1, pos_wavepeaks_indices])
    average_negative_amplitude = np.average(wave_filtered[1, neg_wavepeaks_indices])
    average_amplitude = np.average([average_positive_amplitude, -average_negative_amplitude])
    print(f'\nAverage wave height: {average_amplitude} m')

    # Combine positive and negative peaks into one array
    all_peaks_indices = np.concatenate((pos_wavepeaks_indices, neg_wavepeaks_indices))
    all_peaks_indices.sort()
    all_wavepeaks_filtered = wave_filtered[:, all_peaks_indices]

    # Plot wave history on probe=location
    pop.makeplot(f'Radiated wave at x={(location + 1) * Ldeep:.2f} m',
                 [wave_history[0], all_wavepeaks_filtered[0]], 
                 [wave_history[1], all_wavepeaks_filtered[1]], 
                 'Time (s)', 'Amplitude',
                 ['radiated wave (m)', 'max wave amplitude (m)'], 
                 folder_path, 'wave')


    ##################################################################################################################
    # Force treatment
    ##################################################################################################################
    
    n_periods = int(np.floor((time_truncated[-1] - time_truncated[0]) / T))

    # Calculate the exact time span to cover these periods
    total_time = n_periods * T
    start_time = time_truncated[-1] - total_time
    start_index = np.searchsorted(time_truncated, start_time, side='right')

    # Truncate data based on a number N periods
    time_truncated_n_periods = time_truncated[start_index:]
    forceY_truncated_n_periods = forceY_truncated[start_index:]

    # Smooth the force data
    window_length = 51  # Ensure this is appropriate for your data
    poly_order = 3
    forceY_filtered_n_periods = signal.savgol_filter(forceY_truncated_n_periods, window_length, poly_order)


    ##################################################################################################################
    # Using time-domain integration (Fourier series coefficients) to calculate force amplitude and phase lag
    ##################################################################################################################

    a0 = (1 / (n_periods * T)) * integrate.simpson(y=forceY_filtered_n_periods, x=time_truncated_n_periods) # mean force (buoyancy)

    # In-phase (cosine) and out-of-phase (sine) integration    
    a1 = (2 / (n_periods * T)) * integrate.simpson(y=forceY_filtered_n_periods * np.sin(w * time_truncated_n_periods), x=time_truncated_n_periods)
    b1 = (2 / (n_periods * T)) * integrate.simpson(y=forceY_filtered_n_periods * np.cos(w * time_truncated_n_periods), x=time_truncated_n_periods)

    # Calculate the amplitude and phase of the force
    Fa_amplitude = np.sqrt(a1**2 + b1**2)
    phase1 = np.arctan2(b1, a1)

    # Ensure phase1 is in the correct range (0 to 2π)
    if phase1 < 0:
        phase1 += 2 * np.pi

    ##################################################################################################################
    # Calculate the hydrodynamic coefficients using reconstructed force
    ##################################################################################################################

    # Reconstruct the force using the calculated components
    force_reconstructed_n_periods = a1 * np.cos(w * time_truncated_n_periods) + b1 * np.sin(w * time_truncated_n_periods)

    print(f"\nForce Amplitude: {Fa_amplitude} N")

    print(f'\nRestoring coefficient: {restoringCoeff}', f'\nBuoyancy: {a0} N', "\n")
    added_mass = (restoringCoeff - a1 / motionAmp) / w**2 - rho*np.pi*R**2/2*Z
    damping = b1 / (motionAmp * w)
    print(f"Number of periods used: {n_periods}")

    #added mass and damping using 
    print(f"Added mass coefficient: {added_mass/(rho*np.pi*R**2)}")
    print(f"Damping coefficient: {damping/(rho*np.pi*R**2*w)}")

    coeffs = pop.UzunogluMethod(phase1, Fa_amplitude, motionAmp, w)
    a = coeffs.addedmass
    b = coeffs.damping
    print(f"Added mass coefficient: {a/(rho*np.pi*R**2)}")
    print(f"Damping coefficient: {b/(rho*np.pi*R**2*w)}")   

    pop.makeplot(title='Vertical force on the cylinder',
                 x=[time_truncated, time_truncated_n_periods, time_truncated_n_periods], 
                 y=[forceY_truncated-a0, forceY_filtered_n_periods-a0, force_reconstructed_n_periods], 
                 xlabel='Time (s)', 
                 ylabel='Force (N)',
                 label=['Unfiltered force (N)','Smoothed force (N)','Fourier reconstructed force (N)'], 
                 folder_path=folder_path,
                 figurename='peakforces',
                 linetype=['solid', 'solid', '--'],
                 alpha=[0.8, 1, 1])

    

    ##################################################################################################################
    # Calculate the hydrodynamic coefficients using radiated wave - VUGTS (wave damping)
    ##################################################################################################################

    pop.LinearCoefficients(time[min_truncate_index:], forceY[min_truncate_index:], motionAmp, w, draft, folder_path, rho)
    pop.LinearCoefficients(time_truncated_n_periods, forceY_filtered_n_periods, motionAmp, w, draft, folder_path, rho)    

    # Plot forces
    pop.makeplot(title='Vertical force on the cylinder',
                 x=[time[min_truncate_index:], time_truncated_n_periods], 
                 y=[forceY[min_truncate_index:]-a0, forceY_filtered_n_periods-a0], 
                 xlabel='Time (s)', 
                 ylabel='Force (N)',
                 label=['Unfiltered force (N)','Smoothed force (N)'], 
                 folder_path=folder_path,
                 figurename='periodtruncatedforces',
                 linetype=['solid', 'solid'],
                 alpha=[0.8, 1])