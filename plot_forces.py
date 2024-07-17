#!/usr/bin/python

import os
import sys
import math
import popFOAM as pop
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal 
from scipy import integrate
import hydrocoeffs as hc


grid = "H1.00_predictorON_t100_L0.25"#/RefinementStudy/meshRefinement/wallFunc/interfaceCompression/H1.00_Coarsest_InterfaceCompression"
folder_path = "/media/joaofn/nvme-WD/"+grid+"/"
forces_path = folder_path+"postProcessing/forces/0/"
forces_file = "forces.dat"

if __name__ == "__main__":

    if not os.path.isfile(forces_path+forces_file):
        print("Forces file not found at ", forces_path)
        print("Be sure that the case has been run and you have the right directory!")
        print("Exiting.")
        sys.exit()

    ##################################################################################################################
    # Initialization
    ##################################################################################################################
    
    # Definition of the plots color palette
    color_palette = {
        'color1': '#9E91F2',                # Cold Lips 
        'color2': '#5C548C',                # Purple Corallite
        'color3': '#ABA0F2',                # Dull Lavender
        'color4': '#1A1926',                # Coarse Wool
        'color5': 'orange', 
        'background_color': '#ffffff',      # White Background
        'grid_color': '#F2F2F2',            # Bleached Silk Grid Lines
        'text_color': '#333333',            # Dark Gray Text
        'title_color': '#333333'}           # Dark Gray Title
    
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

    # Check for timestep consistency
    inconsistent_steps = pop.check_time_step_consistency(time, tolerance=1e-6)

    if not inconsistent_steps:
        print("Time step is constant")
    else:
        print("Time step is not constant at the following indices and values:")
        for idx, val in inconsistent_steps:
            print(f"Index {idx}: Time difference {val}")
        #sys.exit()

    # Full motions calculation
    full_motion_signal = [((t / ramp) if t < ramp else 1) * motionAmp * np.sin(w * t) for t in time]
    full_velocity_signal = [((t / ramp) if t < ramp else 1) * w * motionAmp * np.cos(w * t) for t in time]
    full_acceleration_signal = [-((t / ramp) if t < ramp else 1) * w**2 * motionAmp * np.sin(w * t) for t in time]

    # Plot motions
    plt.figure(1, figsize=(12, 8), facecolor=color_palette['background_color'])
    plt.title('Motion time series')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.plot(time, full_motion_signal, color=color_palette['color1'], label='motion (m)')
    plt.plot(time, full_velocity_signal, color=color_palette['color4'], label='velocity (m/s)')
    plt.plot(time, full_acceleration_signal, color=color_palette['color2'], label='acceleration (m/s^2)')
    y_max = max(np.max(full_motion_signal), np.max(full_velocity_signal), np.max(full_acceleration_signal))
    yscale = 1.5                                # Adjust the limits as necessary
    plt.ylim(-yscale * y_max, yscale * y_max)  
    plt.tight_layout(rect=[0, 0, 1, 1])         
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=12)
    plt.grid(True, color=color_palette['grid_color'])
    plt.xticks(color=color_palette['text_color'])
    plt.yticks(color=color_palette['text_color'])
    plt.savefig(folder_path+"motion.pdf", dpi=300, format="pdf")
    plt.close(1)

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
    plt.figure(2, figsize=(12, 8), facecolor=color_palette['background_color'])
    plt.title('Truncated motion time series')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.plot(time_truncated, motion_signal, color=color_palette['color1'], label='motion (m)')
    plt.plot(time_truncated, velocity_signal, color=color_palette['color4'], label='velocity (m/s)')
    plt.plot(time_truncated, acceleration_signal, color=color_palette['color2'], label='acceleration (m/s^2)')
    y_max = max(np.max(full_motion_signal), np.max(full_velocity_signal), np.max(full_acceleration_signal))
    yscale = 1.5                                # Adjust the limits as necessary
    plt.ylim(-yscale * y_max, yscale * y_max)  
    plt.tight_layout(rect=[0, 0, 1, 1])         
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=12)
    plt.grid(True, color=color_palette['grid_color'])
    plt.xticks(color=color_palette['text_color'])
    plt.yticks(color=color_palette['text_color'])
    plt.savefig(folder_path+"truncated_motion.pdf", dpi=300, format="pdf")
    plt.close(2)

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
    print(average_amplitude)

    # Combine positive and negative peaks into one array
    all_peaks_indices = np.concatenate((pos_wavepeaks_indices, neg_wavepeaks_indices))
    all_peaks_indices.sort()
    all_wavepeaks_filtered = wave_filtered[:, all_peaks_indices]

    

    # Plot wave history on probe=location
    plt.figure(3, figsize=(12, 8), facecolor=color_palette['background_color'])
    plt.title(f'Radiated wave at {(location + 1) * Ldeep:.2f}m from the cylinder')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (m)')
    plt.plot(wave_history[0], wave_history[1], color=color_palette['color1'], label='radiated wave (m)')
    plt.plot(all_wavepeaks_filtered[0], all_wavepeaks_filtered[1], linestyle='', marker='o', color=color_palette['color4'], label='max wave amplitude (m)')
    y_max = np.max(wave_history[1])
    yscale = 1.5                                # Adjust the limits as necessary
    plt.ylim(-yscale * y_max, yscale * y_max)  
    plt.tight_layout(rect=[0, 0, 1, 1])         
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=12)
    plt.grid(True, color=color_palette['grid_color'])
    plt.xticks(color=color_palette['text_color'])
    plt.yticks(color=color_palette['text_color'])
    plt.savefig(folder_path+"wave.pdf", dpi=300, format="pdf")
    plt.close(3)

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

    a0 = (1 / (n_periods * T)) * integrate.simpson(y=forceY_filtered_n_periods, x=time_truncated_n_periods)
    # In-phase (cosine) and out-of-phase (sine) integration    
    a1 = (2 / (n_periods * T)) * integrate.simpson(y=forceY_filtered_n_periods * np.sin(w * time_truncated_n_periods), x=time_truncated_n_periods)
    b1 = (2 / (n_periods * T)) * integrate.simpson(y=forceY_filtered_n_periods * np.cos(w * time_truncated_n_periods), x=time_truncated_n_periods)

    # Calculate the amplitude and phase of the force
    Fa_amplitude = np.sqrt(a1**2 + b1**2)
    phase1 = np.arctan2(b1, a1)

    # Plot for diagnostic purposes
    plt.figure(figsize=(12, 6))
    plt.plot(time_truncated_n_periods, Fa_amplitude * np.sin(w * time_truncated_n_periods+phase1), label='Force * sin(wt+phi)')
    plt.plot(time_truncated_n_periods, forceY_filtered_n_periods-a0, label='Force filtered')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Force Components for Integration')
    plt.show()


    # Reconstruct the force using the calculated components
    force_reconstructed_n_periods = a1 * np.cos(w * time_truncated_n_periods) + b1 * np.sin(w * time_truncated_n_periods)

    print(f"Force Amplitude: {Fa_amplitude}")
    print(f"a0 (Mean Force Component): {a0}")
    print(f"a1 (In-Phase Force Component): {a1}")
    print(f"b1 (Out-of-Phase Force Component): {b1}")
    print(f"phase1 shift between force and motion is: {phase1}")

    # Calculate added mass and damping coefficients
    added_mass = (restoringCoeff - a1 / motionAmp) / w**2 - rho*np.pi*R**2/2*Z
    damping = b1 / (motionAmp * w)
    print(f"Number of periods used: {n_periods}")
    print(f"Added mass coefficient: {added_mass/(rho*np.pi*draft**2)}")
    print(f"Damping coefficient: {damping/(rho*np.pi*draft**2*w)}")

    coeffs = hc.UzunogluMethod(phase1, Fa_amplitude, motionAmp, w)
    a = coeffs.addedmass
    b = coeffs.damping
    print(f"Added mass coefficient: {a/(rho*np.pi*draft**2)}")
    print(f"Damping coefficient: {b/(rho*np.pi*draft**2*w)}")   


    # Plot forces
    plt.figure(5, figsize=(12, 8), facecolor=color_palette['background_color'])
    plt.title('Vertical force history on the cylinder')
    plt.xlabel('Time (s)')
    plt.ylabel('Fy (N)')
    plt.plot(time_truncated, forceY_truncated-a0, color=color_palette['color1'], alpha=0.8, label='Unfiltered force (N)')
    plt.plot(time_truncated_n_periods, forceY_filtered_n_periods-a0, color=color_palette['color4'], alpha=1, label='Smoothed force (N)')
    plt.plot(time_truncated_n_periods, force_reconstructed_n_periods, linestyle='--', color=color_palette['color2'], alpha=1, label='Fourier reconstructed force (N)')
    #plt.plot(peak_times, peak_forceY, linestyle='', marker='o', color=color_palette['color5'], label='max force (N)')
    y_max = np.max(forceY_truncated)
    yscale = 1.5                                # Adjust the limits as necessary
    #plt.ylim(-yscale * y_max, yscale * y_max)  
    plt.tight_layout(rect=[0, 0, 1, 1])         
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=12)
    plt.grid(True, color=color_palette['grid_color'])
    plt.xticks(color=color_palette['text_color'])
    plt.yticks(color=color_palette['text_color'])
    plt.savefig(folder_path+"peakforces.pdf", dpi=300, format='pdf')
    plt.close(5)


    ##################################################################################################################
    # Calculate the hydrodynamic coefficients using FFT
    ##################################################################################################################

    #F_in, F_out, num_periods, time_analysis, force_analysis = pop.calculate_force_components(time_truncated, forceY_truncated, w)
    


    ##################################################################################################################
    # Calculate the hydrodynamic coefficients using radiated wave - VUGTS (wave damping)
    ##################################################################################################################



    #coeffs2 = hc.JorgeMethod()    
