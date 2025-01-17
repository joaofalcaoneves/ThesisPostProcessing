#!/usr/bin/python
import popTools as pop
import os
import sys
import math
import numpy as np
from scipy import signal 
from scipy import integrate
from pyfiglet import Figlet


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
    print(text.renderText("popFOAM"))

    if not os.path.isfile(forces_path+forces_file):
        print("Forces file not found at ", forces_path)
        print("Be sure that the case has been run and you have the right directory!")
        print("Exiting...\n\n")
        sys.exit()
    else:
        print("\n\nCASE: " + grid)
        #-----------------------------------------------------------------------------------------------------------------               
        #region Initialization
        time = np.array([])
        forceX = np.array([])
        forceY = np.array([])

        time, forceX, forceY, _ = pop.createForceFile(forces_path+forces_file)
        
        pop.yplus(folder_path, 'floatingObj')
        
        mass = 0                            # mass of cylinder
        g = 9.81                            # acceleration of gravity
        rho = 998.2                         # water density -> be sure to put same as in the simulation!
        draft = 5                           # draft
        motion_amplitude = 0.5                     # motion amplitude
        omega_prime = caseDict[case]             # normalized radial frequency
        omega = np.sqrt(omega_prime * g / draft)     # radial frequency    
        velocity_amplitude = motion_amplitude * omega              # velocity amplitude
        acceleration_amplitude = velocity_amplitude * omega               # acceleration amplitude
        T = 2.0 * math.pi / omega               # period
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
        #endregion
        #-----------------------------------------------------------------------------------------------------------------


        #-----------------------------------------------------------------------------------------------------------------
        #region Time treatment
        try:
            # Check for timestep consistency
            tolerance = 1e-5
            timestep, inconsistent_steps = pop.check_time_step_consistency(time, tolerance=tolerance)
            
            if not inconsistent_steps:
                print(f"\nConst timestep: {timestep} +/- {tolerance}s")
            else:
                print("\nTime step not const at the following indices and values:")
                for idx, val in inconsistent_steps:
                    print(f"Index {idx}: Timestep {val}")
                print(f"\nAvg timestep is: {timestep}s")
        except Exception as e:
            print(f"\nAn error occurred: {e}")        

        min_truncate_index = np.searchsorted(time, truncMin)
        max_truncate_index = np.searchsorted(time, truncMax)
        time_truncated = time[min_truncate_index:max_truncate_index]
        #endregion
        #-----------------------------------------------------------------------------------------------------------------


        #-----------------------------------------------------------------------------------------------------------------        
        #region Motion treatment
        full_motion_signal = np.array([((t / ramp) if t < ramp else 1) * motion_amplitude * np.sin(omega * t) for t in time])
        full_velocity_signal = np.array([((t / ramp) if t < ramp else 1) * omega * motion_amplitude * np.cos(omega * t) for t in time])
        full_acceleration_signal = np.array([-((t / ramp) if t < ramp else 1) * omega**2 * motion_amplitude * np.sin(omega * t) for t in time])

        pop.makeplot(title='Motion',
                    x=time,
                    y=[full_motion_signal, full_velocity_signal, full_acceleration_signal], 
                    xlabel='Time (s)',
                    ylabel='Amplitude',
                    label=['motion (m)', 'velocity (m/s)', 'acceleration (m/s^2)'], 
                    folder_path=folder_path, 
                    figurename='motion')

        # Motion truncation    
        motion_signal = np.array([((t / ramp) if t < ramp else 1) * motion_amplitude * np.sin(omega * t) for t in time_truncated])
        velocity_signal = np.array([((t / ramp) if t < ramp else 1) * omega * motion_amplitude * np.cos(omega * t) for t in time_truncated])
        acceleration_signal = np.array([-((t / ramp) if t < ramp else 1) * omega**2 * motion_amplitude * np.sin(omega * t) for t in time_truncated])    
        
        # Plot motions
        pop.makeplot('Truncated Motion',
                    time_truncated, [motion_signal, velocity_signal, acceleration_signal], 
                    'Time (s)', 'Amplitude',
                    ['motion (m)', 'velocity (m/s)', 'acceleration (m/s^2)'], 
                    folder_path, 'truncated_motion')
        #endregion
        #----------------------------------------------------------------------------------------------------------------- 


        #-----------------------------------------------------------------------------------------------------------------       
        #region Freesurface treatment
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
        average_wave_amplitude = np.average([average_positive_amplitude, average_negative_amplitude])
        print(f'\nAvg wave amplitude: {average_wave_amplitude}m')

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
        #endregion
        #----------------------------------------------------------------------------------------------------------------


        #-----------------------------------------------------------------------------------------------------------------       
        #region Force treatment

        # Force truncation
        forceX_truncated = np.array(forceX[min_truncate_index:max_truncate_index])
        forceY_truncated = np.array(forceY[min_truncate_index:max_truncate_index]) 
        
        # remove restoring force (assuming constant floating plane area)
        #forceY_truncated -= restoringCoeff * np.array(motion_signal)

        # remove avg buoancy 
        forceY_truncated_XX_symetric = forceY_truncated - np.average(forceY_truncated)

        n_periods = int(np.floor((time_truncated[-1] - time_truncated[0]) / T))

        # Calculate the exact time span to cover these periods
        total_time = n_periods * T
        start_time = time_truncated[-1] - total_time
        start_index = np.searchsorted(time_truncated, start_time, side='right')

        # Truncate data based on a number N periods
        time_truncated_n_periods = time_truncated[start_index:]
        forceY_truncated_n_periods = forceY_truncated[start_index:]
        forceY_truncated_n_periods_XX_symetric = forceY_truncated_n_periods - np.average(forceY_truncated_n_periods)
        #endregion
        #-----------------------------------------------------------------------------------------------------------------


        #-----------------------------------------------------------------------------------------------------------------
        #region Smooth the force data
        #-----------------------------------------------------------------------------------------------------------------

        window_length = 21  # Ensure this is appropriate for your data
        poly_order = 3
        forceY_filtered_n_periods = signal.savgol_filter(forceY_truncated_n_periods, window_length, poly_order)
        forceY_filtered_n_periods_XX_symetric = forceY_filtered_n_periods - np.average(forceY_filtered_n_periods)
        #endregion
        #-----------------------------------------------------------------------------------------------------------------


        #-----------------------------------------------------------------------------------------------------------------
        #region Calc up-zero crossings w/ smoothed force
        #-----------------------------------------------------------------------------------------------------------------

        # Align motion time to match force time range
        aligned_motion_indices = (time_truncated >= time_truncated_n_periods[0]) & (time_truncated <= time_truncated_n_periods[-1])
        aligned_motion_times = time_truncated[aligned_motion_indices]
        aligned_motion_signal = motion_signal[aligned_motion_indices]

        # Get upward zero-crossings for motion and force
        # Find zero-crossings for the aligned motion
        _, _, motion_up_times, motion_up_values = pop.find_zero_crossings(aligned_motion_signal, aligned_motion_times, crossing_type="up")
        _, _, force_up_times, force_up_values = pop.find_zero_crossings(forceY_filtered_n_periods_XX_symetric, time_truncated_n_periods, crossing_type="up")

        # Normalize signals
        force_normalized = forceY_filtered_n_periods_XX_symetric / np.max(np.abs(forceY_filtered_n_periods_XX_symetric))
        motion_normalized = aligned_motion_signal / np.max(np.abs(aligned_motion_signal))

        # Ensure matching lengths for zero-crossings
        num_crossings = min(len(motion_up_times), len(force_up_times))

        # Compute time differences (Δt) - considering force leading
        time_differences = -force_up_times[:num_crossings] + motion_up_times[:num_crossings]

        # Calculate phase lags
        phase_shifts = omega * time_differences

        # Average phase lag
        average_phase_shift = np.mean(phase_shifts)

        # Print results
        #print(f"Phase Lags (radians): {phase_shifts_normalized}")
        print(f"\nAverage Phase Lag (radians): {average_phase_shift}")
        print(f"Average Phase Lag (degrees): {np.degrees(average_phase_shift)}\n")


        # Plot with makeplot
        pop.makeplot('Upzero Crossings', 
            [time_truncated_n_periods, force_up_times, aligned_motion_times, motion_up_times], 
            [force_normalized, force_up_values, motion_normalized, motion_up_values], 
            'time', 'force', 
            ['force', 'force up-zero cross', 'motion', 'motion up-zero cross'], 
            folder_path, 
            'up_zero_cross',
            marker=['', 'x', '', 'x'], 
            linetype=['solid', 'None', 'solid', 'None'])
        #endregion
        #-----------------------------------------------------------------------------------------------------------------


        #-----------------------------------------------------------------------------------------------------------------
        #region Fit pure sin to force
        #-----------------------------------------------------------------------------------------------------------------
        print("\n-------------------------------------------------")
        print("Fit sine function to force history")               
        print("-------------------------------------------------")        
        
        fit_sin_force, force_amplitude_fit, force_phase_fit, _ = pop.fit_force_sin(time_truncated_n_periods, 
                                                                               forceY_truncated_n_periods_XX_symetric,
                                                                               omega, 
                                                                               phase_shift_guess=average_phase_shift
                                                                            )
        
        #force_amplitude_fourier*np.sin(w*time_truncated_n_periods + force_phase_fourier)
        offshore_fit = pop.OffshoreHydromechanicsMethod(force_phase_fit, force_amplitude_fit + np.average(forceY_truncated_n_periods), motion_amplitude, omega, restoringCoeff, mass)
        added_mass_fit = offshore_fit.addedmass
        damping_fit = offshore_fit.damping

        print(
            "\nCalculating hydrodynamic coeffs from Offshore Hydromechanics:",
            f"\nAdded mass coeff, u: {round(added_mass_fit)} N.s²/m",
            f"\nDamping coeff, v: {round(damping_fit)} N.s/m",
            f"\nRestoring coeff, k: {round(restoringCoeff)} N/m", 
            f"\nNormalized added mass,u`: {round(pop.normalize(added_mass_fit, draft, rho),4)}",
            f"\nNormalized dammping, v`: {round(pop.normalize(damping_fit, draft, rho, omega, damping=True), 4)}"
        )

        sin_fit = pop.UzunogluMethod(force_phase_fit, force_amplitude_fit, motion_amplitude, omega, mass)
        added_mass_uzunoglu_fit = sin_fit.addedmass
        damping_uzunoglu_fit = sin_fit.damping    
        print(
            "\nCalculated hydrodynamic coeff using Uzunoglu method:",
            f"\nAdded mass coeff, u: {round(added_mass_uzunoglu_fit)} N.s²/m",
            f"\nDamping coeff, v: {round(damping_uzunoglu_fit)} N.s/m",
            f"\nNormalized added mass,u`: {round(pop.normalize(added_mass_uzunoglu_fit, draft, rho),4)}",
            f"\nNormalized dammping, v`: {round(pop.normalize(damping_uzunoglu_fit, draft, rho, omega, damping=True), 4)}"
        )            

        #endregion
        #-----------------------------------------------------------------------------------------------------------------


        #-----------------------------------------------------------------------------------------------------------------        
        #region Time-domain integration (Fourier series coefficients) to calculate force amplitude and phase lag
        print("\n-------------------------------------------------")
        print("Time-domain integration of force (Fourier series)")               
        print("-------------------------------------------------") 

        a0 = (1 / (n_periods * T)) * integrate.simpson(y=forceY_filtered_n_periods_XX_symetric, x=time_truncated_n_periods) # mean force (buoyancy)

        # In-phase (cosine) and out-of-phase (sine) integration    
        a1 = (2 / (n_periods * T)) * integrate.simpson(y=forceY_filtered_n_periods_XX_symetric * np.sin(omega * time_truncated_n_periods), x=time_truncated_n_periods)
        b1 = (2 / (n_periods * T)) * integrate.simpson(y=forceY_filtered_n_periods_XX_symetric * np.cos(omega * time_truncated_n_periods), x=time_truncated_n_periods)
        
        # Calculate the amplitude and phase of the force
        force_amplitude_fourier = np.sqrt(a1**2 + b1**2)
        force_phase_fourier = np.arctan2(b1, a1)  # Calc and normalize phase to [-π, π]

        print(f"# of cycles: {n_periods}",
            f"\nForce amplitude: {round(force_amplitude_fourier)} N",
            f"\nPhase shift: {round(180*force_phase_fourier/np.pi, 2)}º")
        #endregion
        #-----------------------------------------------------------------------------------------------------------------


        #-----------------------------------------------------------------------------------------------------------------               
        #region Calculate the hydrodynamic coefficients of time-domain integration (Fourier series)
             
        # Reconstruct the force using the calculated components
        force_reconstructed_n_periods = a1 * np.cos(omega * time_truncated_n_periods) + b1 * np.sin(omega * time_truncated_n_periods) 

        offshore_fourier = pop.OffshoreHydromechanicsMethod(force_phase_fourier, force_amplitude_fourier + np.average(forceY_truncated_n_periods), motion_amplitude, omega, restoringCoeff, mass)
        added_mass_fourier = offshore_fourier.addedmass
        damping_fourier = offshore_fourier.damping

        print(
            "\nCalculating hydrodynamic coeffs from Offshore Hydromechanics:",
            f"\nAdded mass coeff, u: {round(added_mass_fourier)} N.s²/m",
            f"\nDamping coeff, v: {round(damping_fourier)} N.s/m",
            f"\nRestoring coeff, k: {round(restoringCoeff)} N/m", 
            f"\nNormalized added mass,u`: {round(pop.normalize(added_mass_fourier, draft, rho),4)}",
            f"\nNormalized dammping, v`: {round(pop.normalize(damping_fourier, draft, rho, omega, damping=True), 4)}"
        )

        uzunoglu_fourier = pop.UzunogluMethod(np.pi - force_phase_fourier, force_amplitude_fourier, motion_amplitude, omega, mass)
        added_mass_uzunoglu_fourier = uzunoglu_fourier.addedmass
        damping_uzunoglu_fourier = uzunoglu_fourier.damping

        print(
            "\nCalculating hydrodynamic coeff using Uzunoglu method:",
            f"\nAdded mass coeff, u: {round(added_mass_uzunoglu_fourier)} N.s²/m",
            f"\nDamping coeff, v: {round(damping_uzunoglu_fourier)} N.s/m",
            f"\nNormalized added mass,u`: {round(pop.normalize(added_mass_uzunoglu_fourier, draft, rho),4)}",
            f"\nNormalized dammping, v`: {round(pop.normalize(damping_uzunoglu_fourier, draft, rho, omega, damping=True), 4)}"
        )

        jorge_fourier = pop.JorgeMethod(acceleration_amplitude, velocity_amplitude, motion_amplitude, force_amplitude_fourier, average_wave_amplitude, omega, rho) 
        added_mass_jorge_fourier = jorge_fourier.addedmass
        damping_jorge_fourier = jorge_fourier.damping

        print(
            "\nCalculating hydrodynamic coeff using Jorge method:",
            f"\nAdded mass coeff, u: {round(added_mass_jorge_fourier)} N.s²/m",
            f"\nDamping coeff, v: {round(damping_jorge_fourier)} N.s/m",
            f"\nNormalized added mass,u`: {round(pop.normalize(added_mass_jorge_fourier, draft, rho),4)}",
            f"\nNormalized dammping, v`: {round(pop.normalize(damping_jorge_fourier, draft, rho, omega, damping=True), 4)}"
        )

        #endregion
        #-----------------------------------------------------------------------------------------------------------------


        #-----------------------------------------------------------------------------------------------------------------   
        #region Calculate the hydrodynamic coefficients using radiated wave - VUGTS (wave damping)

        pop.LinearCoefficients(timestep,forceY_truncated, motion_amplitude, omega, draft)
        
        pop.LinearCoefficients(timestep,forceY_filtered_n_periods, motion_amplitude, omega, draft)    

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

        pop.makeplot(title='Vertical force on the cylinder',
                    x=[time_truncated, time_truncated_n_periods, time_truncated_n_periods, time_truncated_n_periods], 
                    y=[forceY_truncated-a0, forceY_filtered_n_periods-a0, force_reconstructed_n_periods, fit_sin_force], 
                    xlabel='Time (s)', 
                    ylabel='Force (N)',
                    label=['Unfiltered force (N)','Smoothed force (N)','Fourier reconstructed force (N)', 'Predicted force from curve fit (N)'], 
                    folder_path=folder_path,
                    figurename='peakforces',
                    linetype=['solid', 'solid', '--', 'solid'],
                    alpha=[0.8, 1, 1, 1])
        #endregion
        #-----------------------------------------------------------------------------------------------------------------


        #-----------------------------------------------------------------------------------------------------------------        
        #region Print results


        #jorge2 = pop.JorgeMethod(acceleration_amplitude, velocity_amplitude, motion_amplitude, fit_sin_amplitude, average_amplitude, omega, rho)        
        #print(round(4*jorge2.damping/(rho*np.pi*R**2*omega), 4), round(4*jorge2.addedmass/(rho*np.pi*R**2),4))
        #endregion
        #-----------------------------------------------------------------------------------------------------------------


        #-----------------------------------------------------------------------------------------------------------------    