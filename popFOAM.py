from scipy.optimize import root
import numpy as np
import re
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import integrate

class HydroCoeff:
    """
    The HydroCoeff class represents the hydrodynamic coefficients of a system.

    It calculates the damping and added mass based on the phase lag, hydrodynamic force, motion amplitude, and angular frequency.

    Attributes:
        _phaselag (float): The phase lag of the system.
        _hydrodynamicforce (float): The hydrodynamic force acting on the system.
        _motionamplitude (float): The amplitude of motion of the system.
        _w (float): The angular frequency of the system.
        _damping (float): The calculated damping of the system.
        _addedmass (float): The calculated added mass of the system.
        _restoring_coefficient (float): The calculated restoring coefficient of the system.

    Methods:
        __init__(self, phaselag: float, hydrodynamicforce: float, motionamplitude: float, w: float, rho: float, g: float, Awp: float):
            Initializes the HydroCoeff object with the given phase lag, hydrodynamic force, motion amplitude, angular frequency, rho, g, and Awp.
            It also calculates the damping, added mass, and restoring coefficient.

        calculate_restoring_coefficient(self):
            Calculates the restoring coefficient based on the given parameters.

        calculate_damping(self):
            Calculates the damping based on the hydrodynamic force, phase lag, motion amplitude, angular frequency, and restoring coefficient.

        calculate_added_mass(self):
            Calculates the added mass based on the hydrodynamic force, phase lag, motion amplitude, angular frequency, and restoring coefficient.

        __str__(self):
            Returns a string representation of the HydroCoeff object.

        __repr__(self):
            Returns a string representation of the HydroCoeff object that can be used to recreate the object.

        phaselag(self) -> float:
            Returns the phase lag.

        phaselag(self, value: float):
            Sets the phase lag to the given value and recalculates the damping, added mass, and restoring coefficient.

        hydrodynamicforce(self) -> float:
            Returns the hydrodynamic force.

        hydrodynamicforce(self, value: float):
            Sets the hydrodynamic force to the given value and recalculates the damping, added mass, and restoring coefficient.

        motionamplitude(self) -> float:
            Returns the motion amplitude.

        motionamplitude(self, value: float):
            Sets the motion amplitude to the given value and recalculates the damping, added mass, and restoring coefficient.

        w(self) -> float:
            Returns the angular frequency.

        w(self, value: float):
            Sets the angular frequency to the given value and recalculates the damping, added mass, and restoring coefficient.

        damping(self) -> float:
            Returns the damping.

        addedmass(self) -> float:
            Returns the added mass.

        restoring_coefficient(self) -> float:
            Returns the restoring coefficient.
    """

    def __init__(self, phaselag: float, hydrodynamicforce: float, motionamplitude: float, w: float, Awp: float, rho: float = 998.2, g: float = 9.81, calculaterestoring: bool = True) -> None:
        """
        Initializes the HydroCoeff object with the given phase lag, hydrodynamic force, motion amplitude, angular frequency, rho, g, and Awp.
        It also calculates the damping, added mass, and restoring coefficient.
        """
        self._phaselag = phaselag
        self._hydrodynamicforce = hydrodynamicforce
        self._motionamplitude = motionamplitude
        self._w = w
        self._rho = rho
        self._g = g
        self._Awp = Awp
        self.calculaterestoring = calculaterestoring

        self.calculate_restoring_coefficient()
        self.calculate_damping()
        self.calculate_added_mass()

    def calculate_restoring_coefficient(self):
        """
        Calculates the restoring coefficient based on the given parameters.
        """
        if self.calculaterestoring:
            self._restoring_coefficient = self._rho * self._g * self._Awp
        else:
            self.restoring_coefficient = 0

    def calculate_damping(self):
        """
        Calculates the damping based on the hydrodynamic force, phase lag, motion amplitude, angular frequency, and restoring coefficient.
        """
        self._damping = - (self._hydrodynamicforce - self._restoring_coefficient * self._motionamplitude) * np.sin(self._phaselag) / (self._motionamplitude * self._w)

    def calculate_added_mass(self):
        """
        Calculates the added mass based on the hydrodynamic force, phase lag, motion amplitude, angular frequency, and restoring coefficient.
        """
        self._addedmass = (self._hydrodynamicforce - self._restoring_coefficient * self._motionamplitude) * np.cos(self._phaselag) / (self._motionamplitude * self._w ** 2)

    def __str__(self):
        """
        Returns a string representation of the HydroCoeff object.
        """
        return f"HydroCoeff: phaselag={self._phaselag}, hydrodynamicforce={self._hydrodynamicforce}, motionamplitude={self._motionamplitude}, w={self._w}, damping={self._damping}, addedmass={self._addedmass}, restoring_coefficient={self._restoring_coefficient}"

    def __repr__(self):
        """
        Returns a string representation of the HydroCoeff object that can be used to recreate the object.
        """
        return f"HydroCoeff(phaselag={self._phaselag}, hydrodynamicforce={self._hydrodynamicforce}, motionamplitude={self._motionamplitude}, w={self._w}, restoring_coefficient={self._restoring_coefficient})"

    @property
    def phaselag(self) -> float:
        """
        Returns the phase lag.
        """
        return self._phaselag

    @phaselag.setter
    def phaselag(self, value: float):
        """
        Sets the phase lag to the given value and recalculates the damping, added mass, and restoring coefficient.
        """
        self._phaselag = value
        self.calculate_restoring_coefficient()
        self.calculate_damping()
        self.calculate_added_mass()

    @property
    def hydrodynamicforce(self) -> float:
        """
        Returns the hydrodynamic force.
        """
        return self._hydrodynamicforce

    @hydrodynamicforce.setter
    def hydrodynamicforce(self, value: float):
        """
        Sets the hydrodynamic force to the given value and recalculates the damping, added mass, and restoring coefficient.
        """
        self._hydrodynamicforce = value
        self.calculate_restoring_coefficient()
        self.calculate_damping()
        self.calculate_added_mass()

    @property
    def motionamplitude(self) -> float:
        """
        Returns the motion amplitude.
        """
        return self._motionamplitude

    @motionamplitude.setter
    def motionamplitude(self, value: float):
        """
        Sets the motion amplitude to the given value and recalculates the damping, added mass, and restoring coefficient.
        """
        self._motionamplitude = value
        self.calculate_restoring_coefficient()
        self.calculate_damping()
        self.calculate_added_mass()

    @property
    def w(self) -> float:
        """
        Returns the angular frequency.
        """
        return self._w

    @w.setter
    def w(self, value: float):
        """
        Sets the angular frequency to the given value and recalculates the damping, added mass, and restoring coefficient.
        """
        self._w = value
        self.calculate_restoring_coefficient()
        self.calculate_damping()
        self.calculate_added_mass()

    @property
    def damping(self) -> float:
        """
        Returns the damping.
        """
        return self._damping

    @property
    def addedmass(self) -> float:
        """
        Returns the added mass.
        """
        return self._addedmass

    @property
    def restoring_coefficient(self) -> float:
        """
        Returns the restoring coefficient.
        """
        return self._restoring_coefficient
    

class RadiatedWave:
    """
    The `RadiatedWave` class represents a radiated wave and provides methods to calculate the free surface elevation.
    """

    DEFAULT_PROBE = 1

    def __init__(self, waveperiod: float, mainfolderpath: str):
        """
        Initializes a `RadiatedWave` object with the given wave period and main folder path.

        Args:
            waveperiod (float): The wave period of the radiated wave.
            mainfolderpath (str): The path to the main folder containing the wave data.

        Raises:
            ValueError: If waveperiod is not a positive number.
            ValueError: If mainfolderpath is not a valid directory path.
        """
        if waveperiod <= 0:
            raise ValueError("waveperiod must be a positive number")
        if not os.path.isdir(mainfolderpath):
            raise ValueError("mainfolderpath must be a valid directory path")
        self.waveperiod = waveperiod
        self.mainfolderpath = mainfolderpath
        self.wave_history = np.array([])
    def freesurfaceelevation(self, probe=DEFAULT_PROBE, relBottom=False):
        """
        Calculates the free surface elevation based on the probe number and whether it is relative to the bottom or not.

        Args:
            probe (int, optional): The probe number. Defaults to DEFAULT_PROBE.
            relBottom (bool, optional): Whether the elevation is relative to the bottom or not. Defaults to False.

        Returns:
            np.array: The calculated free surface elevation.

        Raises:
            FileNotFoundError: If the file is not found.
            ValueError: If there is an error parsing the file.
        """
        root_dir = os.path.join(self.mainfolderpath, "postProcessing", "interfaceHeight", "0")
        file_path = os.path.join(root_dir, "height.dat")
        time = []
        wave = []
        if relBottom:
            i = 0
        else:
            i = 1
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    if not line.startswith('#'):
                        parts = line.split()
                        wave.append(float(parts[1 + probe*2 + i]))
                        time.append(float(parts[0]))
        except FileNotFoundError:
            print("File not found.")
            return
        except ValueError:
            print("Error parsing file.")
            return
        self.wave_history = np.array([time, wave])
        
        return self.wave_history

def process_line(line):
    # Extract time value
    time = float(line.split()[0])

    # Extract the numeric values within parentheses
    float_values = []
    tokens = re.findall(r'\(([^)]+)\)', line)
    for token in tokens:
        # Removing any lingering parentheses and splitting by spaces
        cleaned_values = token.replace('(', '').replace(')', '').split()
        float_values.extend([float(x) for x in cleaned_values])

    # Return time followed by the extracted float values
    return [time] + float_values

def createForceFile(forces_file: str) -> tuple:
    data = []

    with open(forces_file, "r") as datafile:
        for line in datafile:
            if not line.startswith("#"):
                data.append(tuple(process_line(line)))  # Convert to tuple

    # Define the dtype for the structured array
    dtype = [('time', float), 
            ('pressure_x', float), ('pressure_y', float), ('pressure_z', float),
            ('viscous_x', float), ('viscous_y', float), ('viscous_z', float),
            ('pressure_moment_x', float), ('pressure_moment_y', float), ('pressure_moment_z', float),
            ('viscous_moment_x', float), ('viscous_moment_y', float), ('viscous_moment_z', float)]

    # Create a structured NumPy array
    data_array = np.array(data, dtype=dtype)

    # Extract the required data
    time = data_array['time']
    forceX = data_array['pressure_x'] + data_array['viscous_x']
    forceY = data_array['pressure_y'] + data_array['viscous_y']
    forceZ = data_array['pressure_z'] + data_array['viscous_z']

    return time, forceX, forceY, forceZ

def yplus(folderpath, objectname: str):
    
    path = folderpath+"/postProcessing/yPlus/0/yPlus.dat"
    
    time_values = []
    y_plus_minvalues = []
    y_plus_maxvalues = []
    y_plus_avgvalues = []

    with open(path, 'r') as file:
        for line in file:
            if objectname in line:
                data = line.split()
                time = float(data[0])
                y_plus_min = float(data[2])
                y_plus_max = float(data[3])
                y_plus_avg = float(data[4])
                time_values.append(time)
                y_plus_minvalues.append(y_plus_min)
                y_plus_maxvalues.append(y_plus_max)
                y_plus_avgvalues.append(y_plus_avg)

    plt.plot(time_values, y_plus_minvalues, label='minimum')
    plt.plot(time_values, y_plus_maxvalues, label='maximum')
    plt.plot(time_values, y_plus_avgvalues, label='average')
    plt.legend(loc='upper left')
    plt.xlabel('Time')
    plt.ylabel('y+')
    plt.title('y+ values over time for floatingObj')
    plt.savefig(folderpath+"yplus.pdf", dpi=300, format='pdf')


def calculate_force_components(time, force, omega):
    T = 2 * np.pi / omega
    
    # Calculate the number of complete periods in the truncated data
    num_periods = int((time[-1] - time[0]) / T)
    
    # Calculate the end time for the integer number of periods
    end_time = time[0] + num_periods * T
    
    # Find the index corresponding to the end time
    end_index = np.argmax(time >= end_time)
    
    # Truncate the time and force arrays to include only complete periods
    time_truncated = time[:end_index]
    force_truncated = force[:end_index]
    
    # Define the integrands
    in_phase_integrand = lambda t, f: f * np.cos(omega * t)
    out_phase_integrand = lambda t, f: f * np.sin(omega * t)
    
    # Perform the numerical integration
    F_in = (2/(num_periods*T)) * integrate.trapz(in_phase_integrand(time_truncated, force_truncated), time_truncated)
    F_out = (2/(num_periods*T)) * integrate.trapz(out_phase_integrand(time_truncated, force_truncated), time_truncated)
    
    return F_in, F_out, num_periods, time_truncated, force_truncated

def find_zero_crossings(force, time):
    zero_crossings = np.where(np.diff(np.sign(force)) > 0)[0]
    zero_crossing_times = time[zero_crossings]
    return zero_crossings, zero_crossing_times

def restoringCoefficient(XPoints, YPoints, waterlines, rho=998.2, scale=1000, g=9.81, degree=5, initial_guess=5000):
    """
    Calculates the length of the waterline based on the given X and Y points of the shape outline and the desired waterline height.

    Args:
        XPoints (list): The X coordinates of the data points of the shape outline.
        YPoints (list): The Y coordinates of the data points of the shape outline.
        waterline (float): The desired height of the waterline.
        poly_degree (int, optional): The degree of the polynomial curve to fit. Defaults to 2.
        initial_guess (float, optional): The initial guess for the root finding algorithm. Defaults to 0.

    Returns:
        float: The length of the waterline based on the given X and Y points and the desired waterline height.
    """
    # Fit a polynomial curve to the X and Y points
    coefficients = np.polyfit(XPoints, YPoints, degree)
    polynomial = np.poly1d(coefficients)

    def equation(x):
        return polynomial(x) - waterline
    
    restoringCoefficients = np.array([])
    for waterline in waterlines:
        x_solution = root(equation, initial_guess, method ='lm')
        restoringCoefficients = np.append(restoringCoefficients, 2 * x_solution.x[0] / scale * rho * g)
    #print(f'waterline: {waterline} \nAWL: {waterlineLength}\n\n')
    return restoringCoefficients

def hullshape(path='/mnt/Data1/jneves/of10/VerifictionAndValidation_HeaveBatch/', name ='cylinder_shape.txt', print=False):
    """
    This function reads data from a file, filters it, and plots it using seaborn scatterplot.
    
    Args:
    path (str): The path to the file.
    name (str): The name of the file.
    
    Returns:
    pandas.DataFrame: The filtered data.
    """
    file = path+name

    data = pd.read_csv(file, delimiter=',', header=None, names=['X', 'Y', 'Z'],)

    filtered_data = data[data['X'] > 0]
    filtered_data = filtered_data.sort_values(by='Y')
    if print:
        plt.figure()

        plt.axis('equal')
        plt.xlim(min(filtered_data['X']), max(filtered_data['X']) * 1.1)  # Assuming you want to see a bit beyond the maximum X value
        plt.ylim(min(filtered_data['Y']), max(filtered_data['Y']) * 1.1) 

        #sns.scatterplot(data=filtered_data, x='X', y='Y')

        plt.savefig(file + '.pdf', dpi=300, format='pdf')
        plt.close() 
    return filtered_data

def check_time_step_consistency(time_array, tolerance=1e-6):
    # Calculate the first time difference
    first_diff = time_array[1] - time_array[0]
    
    # Initialize a list to store indices and values of inconsistent time steps
    inconsistent_steps = []

    # Loop through the array and check each time difference
    for i in range(2, len(time_array)):
        current_diff = time_array[i] - time_array[i-1]
        if not np.isclose(current_diff, first_diff, atol=tolerance):
            inconsistent_steps.append((i, current_diff))

    return inconsistent_steps



normalized_frequency_Yeung = [0.08782936010037641, 0.2823086574654956, 0.49560853199498117, 0.6932245922208281, 0.8971141781681305, 1.1041405269761606, 1.2547051442910917, 1.4993726474278546, 1.750313676286073, 2.0984943538268506]
yeung_damping_normalized = [1.5172413793103448, 1.1282758620689655, 0.786206896551724, 0.6013793103448275, 0.46344827586206894, 0.336551724137931, 0.2786206896551724, 0.21241379310344827, 0.15724137931034482, 0.11310344827586206]
yeung_addedMass_normalized = [1.2882758620689654, 0.7834482758620689, 0.6179310344827585, 0.5682758620689655, 0.5903448275862069, 0.6041379310344828, 0.6344827586206896, 0.6537931034482758, 0.6979310344827586, 0.7337931034482758]

normalized_frequency_damping_Sutulo = [0.10664993726474278, 0.266624843161857, 0.3575909661229611, 0.4767879548306148, 0.6273525721455459, 0.8155583437892095, 0.9849435382685069, 1.1982434127979924, 1.4084065244667503, 1.6750313676286073, 1.910288582183187, 2.2302383939774155, 2.4843161856963616]
normalized_damping_Sutulo = [1.5917241379310343, 1.2634482758620689, 1.089655172413793, 0.9020689655172414, 0.7117241379310344, 0.5268965517241379, 0.41379310344827586, 0.30344827586206896, 0.22620689655172413, 0.15724137931034482, 0.1186206896551724, 0.08551724137931034, 0.06620689655172413]

normalized_frequency_addedMass_Sutulo= [0.10351317440401506, 0.1599749058971142, 0.23212045169385195, 0.29799247176913424, 0.3983688833124216, 0.5740276035131744, 0.8124215809284818, 1.0100376411543288, 1.3393977415307403, 1.6405269761606023, 1.932245922208281, 2.22396486825596, 2.487452948557089]
normalized_addedMass_Sutulo = [1.246896551724138, 1.0786206896551724, 0.8855172413793103, 0.7806896551724137, 0.6868965517241379, 0.5958620689655172, 0.5682758620689655, 0.5793103448275861, 0.6317241379310344, 0.6786206896551724, 0.7227586206896551, 0.7586206896551724, 0.7806896551724137]