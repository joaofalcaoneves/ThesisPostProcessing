import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def ForceExtraction(starfile, force_name, width, height, length, period, fig_label, velocity_amp = 0.1, time_var_name="Time[s]", plot_fig=True):
    """Extract force coefficients for S=Asin(wt) motion

    Args:
        starfile (str): input file
        force_name (str): which force to analyze
        width (float): width of the pontoon, used for normalization
        height (float): height of the pontoon, used for normalization
        length (float): total length of the pontoon
        period (float): forced oscillation period
        fig_label (str): figure output name
        velocity_amp (float, optional): velocity amplitude Defaults to 0.1.
        time_var_name (str, optional): name for the time series. Defaults to "Time[s]".
        plot_fig (bool, optional): flag for plotting the force comparison. Defaults to True.

    Returns:
        res_lsq.x : coefficients from least square
    """

    try:
        stardf = pd.read_csv(starfile, sep=None, engine="python")
    except:
        stardf = pd.read_csv(starfile, sep=" ")
    time = stardf[time_var_name].to_numpy()[:]

    # By default only use the second half of the data
    start_idx = len(time)//2
    time = time[start_idx:]

    time_step = time[1]-time[0]
    force = stardf[force_name].to_numpy()[start_idx:]/length # divided by the section length

    # forced ocillation parameters
    # velocity = amp *sin (2*pi*freq*time)
    amp = velocity_amp # m/s
    freq = 1/period
    omega = 2*np.pi/period
    # acceleration amp
    accel_amp = amp*omega
    # Block length
    N = len(force)
    fourier_coeff = scipy.fft.fft(force)[1:len(time)//2]
    fourier_coeff = fourier_coeff* np.exp(-1j*2*np.pi*freq*time[0])
    a0 = scipy.fft.fft(force)[0]*2/N
    fft_freq = scipy.fft.fftfreq(len(time), d=time_step)[1:len(time)//2]
    real_part = np.real(fourier_coeff)*2/N
    imag_part = np.imag(fourier_coeff)*2/N

    # Filter
    df_fft= fft_freq[1] - fft_freq[0] # frequency resolution
    # Take the imaginary parts as the coefficients for sine terms  
    sine_term = - imag_part[[int(freq//df_fft)-1, 2*int(freq//df_fft)-1, 3*int(freq//df_fft)-1, 4*int(freq//df_fft)-1, 5*int(freq//df_fft)-1]]
    # Added mass coefficient
    ca = sine_term[0]/accel_amp/(997.2*width*height) # normalized by rho*a*b
    # Added mass force
    Fa = sine_term[0] * np.sin(omega*time)

    # Quadratic
    # Take the real parts as the coefficients for cosine terms
    cd = real_part[[int(freq//df_fft)-1, 3*int(freq//df_fft)-1, 5*int(freq//df_fft)-1, 7*int(freq//df_fft)-1, 9*int(freq//df_fft)-1, 11*int(freq//df_fft)-1]]
    # Theoretical ratio for quadratic damping
    cdq0 = - cd[0]/(0.5*8/3/np.pi*997.2*amp**2*width)

    # reconstruct force
    fit_force_1 = - cdq0*0.5*997.2*amp**2*width*np.cos(omega*time)*np.abs(np.cos(omega*time))

    # Least square
    # Use FFT results as initial guess
    res_lsq = least_squares(LeastSquareResidual, np.array([2, cdq0]), args=(time, force-a0/2, amp, width, height, omega))
    res_lsq.x = np.real(res_lsq.x)
    fit_force_lsq = - np.real(res_lsq.x[1])*0.5*997.2*amp**2*width*np.cos(omega*time)*np.abs(np.cos(omega*time))

    if plot_fig:

        fig, ax = plt.subplots(1, 2, figsize=[10,4])

        ax[1].plot(time, force - Fa - a0/2, "--k", linewidth=1.0, label="CFD - Fa")
        ax[1].plot(time, fit_force_1, linewidth=1.5, label="Approach 1 with Ca={:.3f} and Cdq={:.3f}".format(ca, cdq0))
        ax[1].plot(time, fit_force_lsq, linewidth=1.5, label="LSQ with Ca={:.3f} and Cdq={:.3f}".format(res_lsq.x[0],res_lsq.x[1]))

        ax[1].set_xlabel("Time [sec]")
        ax[1].set_ylabel("Unit force [N/m]")
        ax[1].legend(loc="lower right", fontsize=10)
        

        ax[0].plot(fft_freq, real_part, label="Real part")
        ax[0].plot(fft_freq, imag_part, label="Imag part")
        ax[0].set_xlabel("Frequency [Hz]")
        ax[0].set_xlim([0.02, 0.6])
        ax[0].legend()

        ax[0].spines["top"].set_color("none")
        ax[0].spines["right"].set_color("none")
        ax[1].spines["top"].set_color("none")
        ax[1].spines["right"].set_color("none")

        plt.tight_layout()
        plt.grid()
        plt.savefig("Comparison_section{}.pdf".format(fig_label), transparent=None, dpi="figure", format="pdf")
        plt.show()

    # By default, return the least square results
    return res_lsq.x

def LeastSquareResidual(coeff, time, force, amp, width, height, omega):
    """Define the residual for least square fit approach

    Args:
        coeff (array): Added mass coefficient and Cdq coefficient in quadratic drag
        time (np array): time sequence [s]
        force (np array): force sequence [N/m, N]
        amp (float): velocity amplitude
        width (float): pontoon width
        omega (float): angular frequency [rad/s]

    Returns:
        res: residual
    """
    FA = coeff[0] * amp * omega * (997.2*width*height) * np.sin(omega*time)
    res = force - (FA) + (coeff[1]*0.5*997.2*amp**2*width*np.cos(omega*time)*np.abs(np.cos(omega*time)))

    return res

if __name__ == "__main__":
    coeff = ForceExtraction(starfile="2024-02-24/NREL_CFD_V_LC07.csv", force_name="Fz[N]", width=12.43, height=7.08, length=18.0, period=20.0, velocity_amp = 2.0*7.08/20.0, fig_label="Focal_KC2_T20", time_var_name="Time[s]", plot_fig=True)
    print("Added mass Ca: ", coeff[0])
    print("Added mass Cd: ", coeff[1])