#!/usr/bin/env python3
import numpy as np
from scipy.fft import fft, fftfreq

class JorgeMethod:

    def __init__(self, acceleration, velocity, motion, force, waveamplitude, w, rho):
        # Passing arguments to instance attributes
        self.motionaacceleration = acceleration
        self.motionvelocity = velocity
        self.force = force
        self.motionamplitude = motion
        self.waveamplitude = waveamplitude
        self.w = w
        self.rho = rho
        # Calculates damping and assigns as instance attribute:
        self.damping = self.rho * (9.81 ** 2) / (self.w ** 3) * ((self.waveamplitude / self.motionamplitude) ** 2)
        # Calculates damping and assigns as instance attribute:
        self.addedmass = (self.force - self.damping * self.motionvelocity) / self.motionaacceleration


class UzunogluMethod:

    def __init__(self, phaselag, hydrodynamicforce, motionamplitude, w):
        # Passing arguments to instance attributes
        self.phaselag = phaselag
        self.hydrodynamicforce = hydrodynamicforce
        self.motionamplitude = motionamplitude
        self.w = w
        # Calculates damping and assigns as instance attribute:
        self.damping = self.hydrodynamicforce * np.sin(self.phaselag) / (self.motionamplitude * self.w)
        # Calculates damping and assigns as instance attribute:
        self.addedmass = self.hydrodynamicforce * np.cos(self.phaselag) / (self.motionamplitude * self.w ** 2)


'''
class OC7:

    def __init__(self, time: np.ndarray, force: np.ndarray, motionAmp: float, omega: float, diameter: float, rho: float = 998.2) -> None:
        # By default only use the second half of the data
        self.time = time
        self.time_step = time[1]-time[0]
        self.force = force
        self.omega = omega
        self.motionAmp = motionAmp
        self.velAmp = omega * self.motionAmp
        self.acelAmp = omega**2 * self.motionAmp
        self.diameter = diameter
        self.rho = rho

        # Block length
        N = len(force)
        
        
        #--------------------------------------------------------------------------------------------------------------------------------------------------
        # Why Skip the First Element?: 
        # The first element of the FFT output (at index 0) represents the DC component,  which is the average
        # value of the signal. It’s often not needed for frequency analysis of oscillatory signals.

        fourier_coeff = scipy.fft.fft(force)[1:len(time)//2] # An array of complex numbers representing the amplitude and phase of each frequency component.
        
        #--------------------------------------------------------------------------------------------------------------------------------------------------
        # Purpose: This line applies a phase shift to the Fourier coefficients. This is done to correct for the phase at the start time of the data.
        # Explanation:
        # -1j represents the imaginary unit.
        # 2 * np.pi * omega * time[0] calculates the phase shift needed.
        # np.exp(-1j * 2 * np.pi * omega * time[0]) computes the complex exponential for the phase correction.

        fourier_coeff = fourier_coeff* np.exp(-1j*2*np.pi*omega*time[0]) # Frequency Shifting (to adjust for phase):

        #--------------------------------------------------------------------------------------------------------------------------------------------------

        a0 = scipy.fft.fft(force)[0]*2/N
        fft_freq = scipy.fft.fftfreq(len(time), d=self.time_step)[1:len(time)//2]
        real_part = np.real(fourier_coeff)*2/N
        imag_part = np.imag(fourier_coeff)*2/N

        # Filter
        df_fft= fft_freq[1] - fft_freq[0] # frequency resolution
 
        #--------------------------------------------------------------------------------------------------------------------------------------------------
        # Sine Term for Damping Coefficient
        # Take the imaginary parts as the coefficients for cosine terms  

        vel_term = - imag_part[[1*int(omega//df_fft)-1, 
                                2*int(omega//df_fft)-1, 
                                3*int(omega//df_fft)-1, 
                                4*int(omega//df_fft)-1, 
                                5*int(omega//df_fft)-1]]
        
        # Damping coefficient
        self.v = vel_term[0]
        
        # Force in phase with vel
        self.F_coswt = vel_term[0] * np.sin(omega*time)

        #--------------------------------------------------------------------------------------------------------------------------------------------------
        # Cosine Term for Added Mass Coefficient
        # cd: Extracts the relevant real parts for different harmonics.

        u = real_part[[1*int(omega//df_fft)-1, 
                        3*int(omega//df_fft)-1, 
                        5*int(omega//df_fft)-1, 
                        7*int(omega//df_fft)-1]] 

        # Theoretical ratio for quadratic damping
        cdq0 = - cd[0]
        # Include the fundamental frequency and even harmonics if necessary
        cd_linear = real_part[[1*int(omega//df_fft)-1, 
                               2*int(omega//df_fft)-1,
                               3*int(omega//df_fft)-1, 
                               4*int(omega//df_fft)-1, 
                               5*int(omega//df_fft)-1, 
                               6*int(omega//df_fft)-1]]

        # Extract coefficients for linear and quadratic damping
        linear_damping_coefficient = cd_linear[0] / (0.5*8/3/np.pi*rho*self.velAmp**2*diameter)       # Linear damping coefficient
'''

class LinearCoefficients:

    def __init__(self, time: np.ndarray, force: np.ndarray, motionAmp: float, omega: float, half_breadth: float, rho: float = 998.2) -> None:
        # By default only use the second half of the data
        self.time = time
        self.time_step = time[1]-time[0]
        self.force = force
        self.omega = omega
        self.motionAmp = motionAmp
        self.velAmp = omega * self.motionAmp
        self.acelAmp = omega**2 * self.motionAmp
        self.half_breadth = half_breadth
        self.rho = rho

        N = len(force)
        fft_result = fft(force)
        frequencies = fftfreq(N, d=self.time_step)

        self.fundamental_index = np.argmax(frequencies>0)
        self.fundamental_frequency = frequencies[self.fundamental_index]
        
        self.real_part = np.real(fft_result[self.fundamental_index])
        self.imaginary_part = np.imag(fft_result[self.fundamental_index])
        self.magnitude = np.abs(fft_result[self.fundamental_index])
        self.phase = np.angle(fft_result[self.fundamental_index])
        
        self.damping = self.real_part / (self.omega * self.motionAmp)
        self.norm_damping = self.damping / (np.pi * self.rho* self.half_breadth**2 * self.omega)
        self.added_mass = -self.imaginary_part / (self.omega**2 * self.motionAmp)
        self.norm_added_mass = self.added_mass / (np.pi * self.rho * self.half_breadth**2)                                       

        print(f"\nFundamental Frequency: {self.fundamental_frequency} Hz")
        print(f"Real Part: {self.real_part}")
        print(f"Imaginary Part: {self.imaginary_part}")
        print(f"\nDamping: {self.damping}")
        print(f"Added mass: {self.added_mass}")
        print(f"\nNormalized damping: {self.norm_damping}")
        print(f"Normalized added mass: {self.norm_added_mass}")
