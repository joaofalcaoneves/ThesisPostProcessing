#!/usr/bin/env python3
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

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
        self.addedmass = -self.hydrodynamicforce * np.cos(self.phaselag) / (self.motionamplitude * self.w ** 2)


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

        # Perform FFT
        N = self.force.size
        fft_result = fft(self.force)
        frequencies = fftfreq(N, d=self.time_step)

        # Calculate the magnitude spectrum
        magnitude = np.abs(fft_result) / N  # Normalized magnitude

        # Select positive frequencies (since the FFT is symmetric)
        positive_frequencies = frequencies > 0
        frequencies = frequencies[positive_frequencies]
        magnitude = magnitude[positive_frequencies]        

        # Find the index of the frequency closest to the excitation frequency
        self.fundamental_index = np.argmax(frequencies>0)
        self.fundamental_frequency = frequencies[self.fundamental_index]

        # Extract the real and imaginary parts at the fundamental frequency
        self.real_part = np.real(fft_result[self.fundamental_index])
        self.imaginary_part = np.imag(fft_result[self.fundamental_index])
        self.magnitude = np.abs(fft_result[self.fundamental_index])
        self.phase = np.angle(fft_result[self.fundamental_index])

        # Calculate damping and added mass        
        self.damping = self.real_part / (self.omega * self.motionAmp)
        self.norm_damping = self.damping / (np.pi * self.rho* self.half_breadth**2 * self.omega)
        self.added_mass = -self.imaginary_part / (self.omega**2 * self.motionAmp)
        self.norm_added_mass = self.added_mass / (np.pi * self.rho * self.half_breadth**2)                                       
        
        # Print results
        print("\n------------------------------------------------\n")
        print(f"\nFundamental Frequency: {self.fundamental_frequency} Hz")
        print(f"Real Part: {self.real_part}")
        print(f"Imaginary Part: {self.imaginary_part}")
        print(f"\nDamping: {self.damping}")
        print(f"Added mass: {self.added_mass}")
        print(f"\nNormalized damping: {self.norm_damping}")
        print(f"Normalized added mass: {self.norm_added_mass}")
        print("\n------------------------------------------------\n\n")

        # Plot the frequency spectrum
        plt.figure(20, figsize=(12, 6))
        plt.plot(frequencies, magnitude)
        plt.title('Frequency Spectrum of Force Data')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.tight_layout()
        plt.show()  # Or save the figure using plt.savefig('spectrum.png', dpi=300)