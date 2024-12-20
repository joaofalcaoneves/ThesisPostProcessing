from scipy.optimize import root
import numpy as np
import re
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import integrate
from scipy.fftpack import fft, fftfreq
import math

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

    def __init__(self, phaselag, hydrodynamicforce, motionamplitude, w, mass):
        # Passing arguments to instance attributes
        self.phaselag = phaselag
        self.hydrodynamicforce = hydrodynamicforce
        self.motionamplitude = motionamplitude
        self.w = w
        self.mass = mass
        # Calculates damping and assigns as instance attribute:
        self.damping = self.hydrodynamicforce * np.sin(self.phaselag) / (self.motionamplitude * self.w)
        # Calculates damping and assigns as instance attribute:
        self.addedmass = -self.hydrodynamicforce * np.cos(self.phaselag) / (self.motionamplitude * self.w ** 2) - self.mass


class LinearCoefficients:

    def __init__(self, timestep: float,  time: np.ndarray, force: np.ndarray, motionAmp: float, omega: float, half_breadth: float, folder_path: str, title: str, rho: float = 998.2) -> None:
        # By default only use the second half of the data
        self.time = time
        self.time_step = timestep
        self.force = force
        self.omega = omega
        self.motionAmp = motionAmp
        self.velAmp = omega * self.motionAmp
        self.acelAmp = omega**2 * self.motionAmp
        self.half_breadth = half_breadth
        self.fig_title = title
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
        self.fundamental_index = np.argmax(frequencies)
        self.fundamental_frequency = frequencies[self.fundamental_index]

        # Extract the real and imaginary parts at the fundamental frequency
        self.real_part = np.real(fft_result[self.fundamental_index])
        self.imaginary_part = np.imag(fft_result[self.fundamental_index])
        self.magnitude = np.abs(fft_result[self.fundamental_index])
        self.phase = np.angle(fft_result[self.fundamental_index])

        # Calculate damping and added mass        
        self.damping = self.real_part / (self.omega * self.motionAmp)
        self.norm_damping = 4 * self.damping / (np.pi * self.rho* self.half_breadth**2 * self.omega)
        self.added_mass = -self.imaginary_part / (self.omega**2 * self.motionAmp)
        self.norm_added_mass = 4 * self.added_mass / (np.pi * self.rho * self.half_breadth**2)                                       
        
        # Print results
        print("\n#######################################################################################")
        print(f"\npopTools.py {type(self).__name__} class output")               
        print(f"------------------------------------------------\n")

        print(f"\nFUNDAMENTAL FREQ: {round(self.fundamental_frequency, 6)} Hz")
        
        print(f"\nREAL PART: {round(self.real_part)}",
              f"\nIMAG PART: {round(self.imaginary_part)}")

        print(f"\nDAMPING: {round(self.damping)} N.s/m",
              f"\nADDED MASS: {round(self.added_mass)} N.s²/m")

        print(f"\nNORMALIZED DAMPING: {round(self.norm_damping, 4)}",
              f"\nNORMALIZED ADDED MASS: {round(self.norm_added_mass,4)}")
        
        print("------------------------------------------------\n")
        print("                  end output                    \n")
        print("------------------------------------------------\n")

        # Plot forces
        makeplot(title='Frequency Spectrum of Force Data',
                    x=[frequencies[frequencies < 1]], 
                    y=[magnitude[frequencies < 1]], 
                    xlabel='Frequency (Hz)', 
                    ylabel='Magnitude',
                    label=[],
                    marker='o', 
                    folder_path=folder_path,
                    figurename=f'{self.fig_title}spectrum')


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



def nondim(coefficient, radius, rho = 998.2):  # only works for cylinder shapes (radius)
    return coefficient / (rho * (math.pi / 2) * radius ** 2)

def makeplot(title: str, x, y, xlabel: str, ylabel: str, label, folder_path: str, figurename: str, marker=None, linetype=None, alpha=None):
    
    # Initialize default values if not provided
    if marker is None:
        marker = ['']  # Default marker if not specified
    if linetype is None:
        linetype = ['solid']  # Default line style if not specified
    if alpha is None:
        alpha = [1]

    # Ensure label is a list or array if it isn't already
    if isinstance(label, str):
        label = [label] * len(y)  # If it's a single string, repeat it for each line
    else:
        label = np.asarray(label)  # Otherwise, convert to array if it's a list
    
    # Definition of the plot's color palette
    color_palette = {
        'color1': '#FFBC42',            # 
        'color2': '#D81159',            # 
        'color3': '#8F2D56',            # 
        'color4': '#218380',            # 
        'color5': '73D2DE',             # 
        'background_color': '#ffffff',  # White Background
        'grid_color': '#d3d3d3',        # Bleached Silk Grid Lines
        'text_color': '#333333',        # Dark Gray Text
        'title_color': '#333333'        # Dark Gray Title
    }

    plt.figure(figsize=(12, 8), facecolor=color_palette['background_color'])
    plt.title(title, color=color_palette['title_color'])
    plt.xlabel(xlabel, color=color_palette['text_color'])
    plt.ylabel(ylabel, color=color_palette['text_color'])
    
    # Get the list of colors for plotting
    colors = list(color_palette.values())[:5]  # Take only the plot colors, excluding non-line colors

    # Prepare x data if it's a single array
    if isinstance(x, np.ndarray) or isinstance(x, list):
        if isinstance(x, np.ndarray) and x.ndim == 1 or isinstance(x, list) and len(x) > 0 and isinstance(x[0], (int, float)):
            # If x is a single array or a list of numbers, check if all y arrays match its length
            if not all(len(y_arr) == len(x) for y_arr in y):
                raise ValueError("When `x` is a single array, it must have the same length as each `y[i]`.")
            # Duplicate `x` for each y if it is valid
            x = [x] * len(y)

    # Loop through each y-array and plot
    for i, _ in enumerate(y):
        # Choose a color from the color palette
        color = colors[i % len(colors)]

        # Get marker and line style for the current line
        marker_style = marker[i] if i < len(marker) else ''
        line_style = linetype[i] if i < len(linetype) else 'solid'
        alpha_style = alpha[i] if i < len(alpha) else 1

        # Plot the line with the chosen color, marker, and line style
        plt.plot(x[i], y[i], color=color, label=label[i] if i < len(label) else f'Line {i+1}', 
                 marker=marker_style, linestyle=line_style, alpha=alpha_style)

    # Adjust y-axis limits
    y_max = max(np.max(y[i]) for i in range(len(y)))
    y_min = min(np.min(y[i]) for i in range(len(y)))
    yscale = 0.25  # Adjust the limits as necessary
    plt.ylim(y_min - abs(yscale * y_min), y_max + abs(yscale * y_max))
    
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=12)
    plt.grid(True, which='both', axis='both', color=color_palette['grid_color'])
    plt.xticks(color=color_palette['text_color'])
    plt.yticks(color=color_palette['text_color'])
    plt.minorticks_on()

    # Save the figure
    save_path = os.path.join(folder_path, figurename + ".pdf")
    plt.savefig(save_path, dpi=300, format="pdf")
    plt.close()

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

    # Use the same x array for all y arrays
    makeplot(title='y+ evolution', 
             x=time_values,  # Single x array
             y=[y_plus_minvalues, y_plus_maxvalues, y_plus_avgvalues],  # List of y arrays
             xlabel='time (s)', 
             ylabel='y+', 
             label=['min', 'max', 'avg'], 
             folder_path=folderpath,
             figurename='yplus')

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

    if inconsistent_steps:
        avg_delta = np.mean([time for _, time in inconsistent_steps])
    else:
        avg_delta = first_diff    

    return avg_delta, inconsistent_steps


