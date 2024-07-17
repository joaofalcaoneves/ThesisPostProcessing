#!/usr/bin/env python3
import numpy as np


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
