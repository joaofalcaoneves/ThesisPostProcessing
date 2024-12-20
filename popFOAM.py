#!/usr/bin/python
import popTools as pop
import os
import sys
import math
import numpy as np
from scipy import signal 
from scipy import integrate
from pyfiglet import Figlet
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from tabulate import tabulate

caseDict = {"H1":0.20, "H2":0.35, "H3":0.5, "H4" :0.75, "H5" :1.0, "H6":1.25,
            "H7":1.50, "H8":1.75, "H9":2.0, "H10":2.25, "H11":2.50}
case = "H5"
case_time = "800"
case_level = "05"
grid = f"{case}/{case}_L{case_level}/{case}_predictorON_t{case_time}_L{case_level}"
folder_path = "/media/joaofn/nvme-WD/"+grid+"/" # "/Users/jneves/Documents/Thesis/Results/"  
forces_path = folder_path+"postProcessing/forces/0/"
forces_file = "forces.dat"

if __name__ == "__main__":
    
    text = Figlet(font='slant')
    print(text.renderText("Let's popFOAM"))

    if not os.path.isfile(forces_path+forces_file):
        print("Forces file not found at ", forces_path)
        print("Be sure that the case has been run and you have the right directory!")
        print("Exiting...\n\n")
        sys.exit()
    else:
        print("\n\nCASE: " + grid)
        
        ##################################################################################################################
        # Initialization
        ##################################################################################################################

        # Var init
        time = np.array([])
        forceX = np.array([])
        forceY = np.array([])

        time, forceX, forceY, _ = pop.createForceFile(forces_path+forces_file)
        
        pop.yplus(folder_path, 'floatingObj')
        
        mass = 0                            # mass of cylinder
        g = 9.81                            # acceleration of gravity
        rho = 998.2                         # water density -> be sure to put same as in the simulation!
        draft = 5                           # draft
        motionAmp = 0.5                     # motion amplitude
        wprime = caseDict[case]             # normalized radial frequency
        w = np.sqrt(wprime * g / draft)     # radial frequency    
        velAmp = motionAmp * w              # velocity amplitude
        accelAmp = velAmp * w               # acceleration amplitude
        T = 2.0 * math.pi / w               # period
        freq = 1 / T                        # frequency    
        Ldeep = g / (2 * math.pi) * (T**2)  # length of wave in deep water
        truncMax = np.max(time)             # max time to analyze
        truncMin = 2 * T                    # min time to analyze
        ramp = 3 * T                        # transient time
        R = draft                           # radius of the cylinder = draft
        Z = 1                               # 2D domain depth in Z    
        Awp = 2 * R * Z                     # waterplane area
        restoringCoeff = Awp * rho * g      # restoring coefficient
        twoL = 2 * Ldeep                    # 2 * length of deep water wave 

        try:
            # Check for timestep consistency
            timestep, inconsistent_steps = pop.check_time_step_consistency(time, tolerance=1e-6)
            
            if not inconsistent_steps:
                print(f"\nCONSTANT TIME STEP: {timestep}s")
            else:
                print("\nTime step not constant at the following indices and values:")
                for idx, val in inconsistent_steps:
                    print(f"Index {idx}: Time difference {val}")
                print(f"\nAvg timestep is: {timestep}s")
        except Exception as e:
            print(f"\nAn error occurred: {e}")

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
        min_truncate_index = np.searchsorted(time, truncMin)
        max_truncate_index = np.searchsorted(time, truncMax)
        time_truncated = time[min_truncate_index:max_truncate_index]

        motion_signal = [((t / ramp) if t < ramp else 1) * motionAmp * np.sin(w * t) for t in time_truncated]
        velocity_signal = [((t / ramp) if t < ramp else 1) * w * motionAmp * np.cos(w * t) for t in time_truncated]
        acceleration_signal = [-((t / ramp) if t < ramp else 1) * w**2 * motionAmp * np.sin(w * t) for t in time_truncated]    
        
        # Force truncation
        forceX_truncated = np.array(forceX[min_truncate_index:max_truncate_index])
        forceY_truncated = np.array(forceY[min_truncate_index:max_truncate_index]) 

        # Plot motions
        pop.makeplot('Truncated Motion',
                    time_truncated, [motion_signal, velocity_signal, acceleration_signal], 
                    'Time (s)', 'Amplitude',
                    ['motion (m)', 'velocity (m/s)', 'acceleration (m/s^2)'], 
                    folder_path, 'truncated_motion')


        ##################################################################################################################
        # Freesurface calculation
        ##################################################################################################################

        location = 1
        radiated_wave = pop.RadiatedWave(waveperiod=T, mainfolderpath=folder_path)
        radiated_wave.freesurfaceelevation(probe=location, relBottom=False)
        wave_history = radiated_wave.wave_history
        
        # Time to truncate wave != truncate min
        minTime = truncMin
        mask = wave_history[0] > minTime
        wave_filtered = wave_history[:, mask]

        # Find wave peaks
        pos_wavepeaks_indices, _ = signal.find_peaks(wave_filtered[1], height=0, threshold=None)
        neg_wavepeaks_indices, _ = signal.find_peaks(-wave_filtered[1], height=0, threshold=None)        
        
        average_positive_amplitude = np.average(wave_filtered[1, pos_wavepeaks_indices])
        average_negative_amplitude = -np.average(wave_filtered[1, neg_wavepeaks_indices])
        average_amplitude = np.average([average_positive_amplitude, average_negative_amplitude])
        print(f'\nAVG WAVE AMPLITUDE: {average_amplitude}m')

        # Combine positive and negative peaks into one array
        all_peaks_indices = np.concatenate((pos_wavepeaks_indices, neg_wavepeaks_indices))
        all_peaks_indices.sort()
        all_wavepeaks_filtered = wave_filtered[:, all_peaks_indices]

        # Plot wave history on probe=location
        pop.makeplot(f'Radiated wave at x={(location + 1) * Ldeep:.2f}m',
                    [wave_history[0], all_wavepeaks_filtered[0]], 
                    [wave_history[1], all_wavepeaks_filtered[1]], 
                    'Time (s)', 'Amplitude',
                    ['radiated wave (m)', 'max wave amplitude (m)'],
                    folder_path, 
                    'wave',
                    marker=['', 'o'],
                    linetype=['solid','None'])


        ##################################################################################################################
        # Force treatment
        ##################################################################################################################
        
        # remove restoring force (assuming constant floating plane area)
        forceY_truncated -= restoringCoeff * np.array(motion_signal)

        n_periods = int(np.floor((time_truncated[-1] - time_truncated[0]) / T))

        # Calculate the exact time span to cover these periods
        total_time = n_periods * T
        start_time = time_truncated[-1] - total_time
        start_index = np.searchsorted(time_truncated, start_time, side='right')

        # Truncate data based on a number N periods
        time_truncated_n_periods = time_truncated[start_index:]
        forceY_truncated_n_periods = forceY_truncated[start_index:]

        # Smooth the force data
        window_length = 21  # Ensure this is appropriate for your data
        poly_order = 3
        forceY_filtered_n_periods = signal.savgol_filter(forceY_truncated_n_periods, window_length, poly_order)


        ##################################################################################################################
        # Using time-domain integration (Fourier series coefficients) to calculate force amplitude and phase lag
        ##################################################################################################################
        print("\n#######################################################################################")
        print("\nUsing time-domain integration (Fourier series coefficients) to \ncalculate force amplitude and phase lag")               
        print("------------------------------------------------\n") 

        a0 = (1 / (n_periods * T)) * integrate.simpson(y=forceY_filtered_n_periods, x=time_truncated_n_periods) # mean force (buoyancy)

        # In-phase (cosine) and out-of-phase (sine) integration    
        a1 = (2 / (n_periods * T)) * integrate.simpson(y=forceY_filtered_n_periods * np.sin(w * time_truncated_n_periods), x=time_truncated_n_periods)
        b1 = (2 / (n_periods * T)) * integrate.simpson(y=forceY_filtered_n_periods * np.cos(w * time_truncated_n_periods), x=time_truncated_n_periods)

        # Calculate the amplitude and phase of the force
        force_amplitude_fourier = np.sqrt(a1**2 + b1**2)
        force_phase_fourier = np.arctan2(b1, a1)

        print(force_phase_fourier)

        # Ensure force_phase_fourier is in the correct range (0 to π/2)
        if force_phase_fourier > np.pi/2:
            force_phase_fourier = np.pi - force_phase_fourier

        print(force_phase_fourier)

        print(f"# OF CYCLES: {n_periods}",
            f"\nFORCE AMPLITUDE: {round(force_amplitude_fourier)} N",
            f"\nFORCE/MOTION PHASE: {round(180*force_phase_fourier/np.pi, 2)}º")
        
        ##################################################################################################################
        # Calculate the hydrodynamic coefficients using reconstructed force
        ##################################################################################################################
        print("\n#######################################################################################")
        print("\nCalculating hydrodynamic coefficients from force Fourier series \ncoefficients")               
        print("------------------------------------------------\n") 
        # Reconstruct the force using the calculated components
        force_reconstructed_n_periods = a1 * np.cos(w * time_truncated_n_periods) + b1 * np.sin(w * time_truncated_n_periods) #force_amplitude_fourier*np.sin(w*time_truncated_n_periods + force_phase_fourier)
        added_mass = (restoringCoeff - (force_amplitude_fourier / motionAmp) * np.cos(force_phase_fourier)) / w**2
        damping = (force_amplitude_fourier * np.sin(force_phase_fourier)) / (motionAmp * w)


        print(f'\nADDED MASS COEFF: {round(added_mass)} N.s²/m',
            f'\nDAMPING COEFF: {round(damping)} N.s/m',
            f'\nRESTORING COEFF: {round(restoringCoeff)} N/m', 
            f'\nBUOYANCY: {round(a0)} N', "\n")


        #added mass and damping using 
        print(f"NORMALIZED ADDED MASS: {round(4*added_mass/(rho*np.pi*R**2),4)}")
        print(f"NORMALIZED DAMPING: {round(4*damping/(rho*np.pi*R**2*w), 4)}")


        ##################################################################################################################
        # Calculate the hydrodynamic coefficients using Uzunuglo method
        ##################################################################################################################
        print("\n#######################################################################################")
        print("\nCalculating hydrodynamic coefficients using force amplitude \nand phase")               
        print("------------------------------------------------\n") 

        coeffs = pop.UzunogluMethod(force_phase_fourier, force_amplitude_fourier, motionAmp, w, mass)
        a = coeffs.addedmass
        b = coeffs.damping
        print(f"NORMALIZED ADDED MASS: {4*a/(rho*np.pi*R**2)}")
        print(f"NORMALIZED DAMPING: {4*b/(rho*np.pi*R**2*w)}")       

        ##################################################################################################################
        # Calculate the hydrodynamic coefficients using radiated wave - VUGTS (wave damping)
        ##################################################################################################################

        pop.LinearCoefficients(timestep, time_truncated, forceY_truncated, motionAmp, w, draft, folder_path, "original_force_", rho)
        pop.LinearCoefficients(timestep, time_truncated_n_periods, forceY_filtered_n_periods, motionAmp, w, draft, folder_path, "filtered_force_", rho)    

        # Plot forces
        pop.makeplot(title='Vertical force on the cylinder',
                    x=[time_truncated, time_truncated_n_periods], 
                    y=[forceY_truncated-a0, forceY_filtered_n_periods-a0], 
                    xlabel='Time (s)', 
                    ylabel='Force (N)',
                    label=['Unfiltered force (N)','Smoothed force (N)'], 
                    folder_path=folder_path,
                    figurename='periodtruncatedforces',
                    linetype=['solid', 'solid'],
                    alpha=[0.8, 1])
            
        def sin_func(t, amplitude, phase):
            return amplitude * np.sin((t * w) + phase)
        
        # Fit cos function to force data
        constants = curve_fit(sin_func, time_truncated_n_periods, forceY_truncated_n_periods)
        force_amplitude_sin_fit = abs(constants[0][0])  # Maximum force amplitude given by cos curve fit
        force_phase_sin_fit = constants[0][1]  # Force phase given by cos curve fit

        # Ensure force_phase_fourier is in the correct range (0 to π/2)
        force_phase_sin_fit = np.arctan2(np.sin(force_phase_sin_fit), np.cos(force_phase_sin_fit))

        if force_phase_sin_fit > np.pi/2:
            force_phase_sin_fit = np.pi - force_phase_sin_fit

        predictedYPressureF = sin_func(time_truncated_n_periods, force_amplitude_sin_fit, force_phase_sin_fit)

        rr = r2_score(forceY_truncated_n_periods, predictedYPressureF)  # R² value for force fit

        print(f"\nFORCE AMPLITUDE: {round(force_amplitude_sin_fit)} N",
            f"\nFORCE/MOTION PHASE: {round(180*force_phase_sin_fit/np.pi, 2)}º",
            f"\nR² score is: {rr}")
        
        old = pop.UzunogluMethod(force_phase_sin_fit, force_amplitude_sin_fit, motionAmp, w, mass)
        print(4*old.addedmass/(rho*np.pi*R**2), 4*old.damping/(rho*np.pi*R**2*w))

        pop.makeplot(title='Vertical force on the cylinder',
                    x=[time_truncated, time_truncated_n_periods, time_truncated_n_periods, time_truncated_n_periods], 
                    y=[forceY_truncated-a0, forceY_filtered_n_periods-a0, force_reconstructed_n_periods, predictedYPressureF], 
                    xlabel='Time (s)', 
                    ylabel='Force (N)',
                    label=['Unfiltered force (N)','Smoothed force (N)','Fourier reconstructed force (N)', 'Predicted force from curve fit (N)'], 
                    folder_path=folder_path,
                    figurename='peakforces',
                    linetype=['solid', 'solid', '--', 'solid'],
                    alpha=[0.8, 1, 1, 1])


        ##################################################################################################################
        # Print results
        #################################################################################################################

        jorge1 = pop.JorgeMethod(accelAmp, velAmp, motionAmp, force_amplitude_fourier, average_amplitude, w, rho) 
        jorge2 = pop.JorgeMethod(accelAmp, velAmp, motionAmp, force_amplitude_sin_fit, average_amplitude, w, rho)        

        print(round(4*jorge1.damping/(rho*np.pi*R**2*w), 4), round(4*jorge1.addedmass/(rho*np.pi*R**2),4))
        print(round(4*jorge2.damping/(rho*np.pi*R**2*w), 4), round(4*jorge2.addedmass/(rho*np.pi*R**2),4))
