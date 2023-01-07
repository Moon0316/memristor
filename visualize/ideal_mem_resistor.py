from plot_memristor_properties import plot_mem_fluctuation
import numpy as np


# Some from the following repo:
# https://github.com/DuttaAbhigyan/Memristor-Simulation-Using-Python/blob/master/models/ideal_memristor.py
class ideal_memristor(object):

    # Initialize the ideal memristor with paramters to be used
    def __init__(self, D=1e-8, w_0=0, R_on=1e3, R_off=16e4,
                 mobility_u=10 ** -14, polarity_n=1):
        self.D = D
        self.w_0 = w_0
        self.R_off = R_off
        self.R_on = R_on
        self.mobility_u = mobility_u
        self.polarity_n = polarity_n
        self.flux_history = 0
        self.resistance = R_off
        self.calculate_fixed_parameters()
        # print(self.D, self.w_0,self.R_off, self.mobility_u)

    # Calculate parameters which are not time varying
    def calculate_fixed_parameters(self):
        self.Q_0 = (self.D ** 2) / (self.mobility_u * self.R_on)
        self.R_delta = self.R_off - self.R_on
        self.R_0 = self.R_on * (self.w_0 / self.D) + self.R_off * (1 - self.w_0 / self.D)
        print(self.R_0)

    # Calculate time variable paramters
    def calculate_time_variable_parameters(self, flux):
        self.flux = self.flux_history + flux
        # Only when q valid to update!
        if self.polarity_n * self.flux <= self.Q_0 * self.R_0 ** 2 / (2 * self.R_delta) * (
                1 - self.R_on ** 2 / self.R_0 ** 2):
            self.drift_factor = (2 * self.polarity_n * self.R_delta * self.flux) / \
                                (self.Q_0 * (self.R_0 ** 2))
        # self.resistance = self.R_0 * ((1 - self.drift_factor) ** 0.5)

    # Calculate current through memristor
    def calculate_current(self, voltage):
        self.current = (voltage / self.R_0) / ((1 - self.drift_factor) ** 0.5)
        self.resistance = self.R_0 * ((1 - self.drift_factor) ** 0.5)

    # Calculate charge through Memristor
    def calculate_charge(self):
        self.charge = ((self.Q_0 * self.R_0) / self.R_delta) * \
                      (1 - (1 - self.drift_factor) ** 0.5) * self.polarity_n

    # Update flux history
    def save_flux_history(self):
        self.flux_history = self.flux

    # Getter functions
    def get_current(self):
        return self.current

    def get_charge(self):
        return self.charge

    def get_flux(self):
        return self.flux

    def get_resistance(self):
        return self.resistance

    def get_conductance(self):
        return 1 / self.resistance

    def simulate_sin_activation(self, amp=1.0, omega=np.pi, sample_num=10000):
        maxtime = 2 * np.pi / omega
        timespace = np.linspace(0, maxtime, sample_num + 1)
        voltage = amp * np.sin(timespace * omega)
        current = voltage / self.R_0 / (
                1 - 2 * self.polarity_n * self.R_delta * amp * (1 - np.cos(timespace * omega)) / (
                self.Q_0 * self.R_0 * self.R_0 * omega)) ** 0.5
        return voltage, current

    def infer(self, voltage, time_step):  # infer one step pf MR
        delta_flux = voltage * time_step
        self.calculate_time_variable_parameters(delta_flux)
        self.calculate_current(voltage)
        self.calculate_charge()
        self.save_flux_history()

    def infer_time_steps(self, voltage: np.ndarray, uni_timestep):  # infer multiple steps
        i_ret = []
        steps = voltage.shape[0]
        for i in range(steps):
            self.infer(voltage[i], uni_timestep)
            i_ret.append(self.get_current())
        return np.array(i_ret)


# End of Ideal Memristor class definition

import matplotlib.pyplot as plt

if __name__ == '__main__':
    MR = ideal_memristor()
    MR.calculate_fixed_parameters()
    eps = 1e-12

    v, i = MR.simulate_sin_activation(omega=np.pi)

    slices = np.abs(v) > eps  # for R=0 bad ones
    v, i = v[slices], i[slices]
    plot_mem_fluctuation(v, i, "Voltage (V)", "Current (A)", "Hysteresis Loop-omega=pi",
                         "ideal-memristor-hys-loop-I-U-default.png")
    plot_mem_fluctuation(v, v / i, "Voltage (V)", "Resistance (Ohm)", "Hysteresis Loop-omega=pi",
                         "ideal-memristor-hys-loop-R-U-default.png")

    # define your desired function here!
    T = 4
    N = 50000
    x = np.linspace(0, T, N + 1)  # eliminate minor error on R
    delta_t = T / N
    v = np.sin(np.pi / 2 * x)
    i = MR.infer_time_steps(v, delta_t)
    print(MR.get_charge())

    slices = np.abs(v) > eps  # for R=0 bad ones
    v, i = v[slices], i[slices]
    plot_mem_fluctuation(v, i, "Voltage (V)", "Current (A)", "Hysteresis Loop-omega=pi/2",
                         "ideal-memristor-hys-loop-I-U-minfreq.png")
    plot_mem_fluctuation(v, v / i, "Voltage (V)", "Resistance (Ohm)", "Hysteresis Loop-omega=pi/2",
                         "ideal-memristor-hys-loop-R-U-minfreq.png")

    # Test the situation under high frequency

    v, i = MR.simulate_sin_activation(omega=np.pi * 100)
    slices = np.abs(v) > eps  # for R=0 bad ones
    v, i = v[slices], i[slices]
    plot_mem_fluctuation(v, i, "Voltage (V)", "Current (A)", "Hysteresis Loop-omega=100pi",
                         "ideal-memristor-hys-loop-I-U-highfreq.png")
    plot_mem_fluctuation(v, v / i, "Voltage (V)", "Resistance (Ohm)", "Hysteresis Loop-omega=100pi",
                         "ideal-memristor-hys-loop-R-U-highfreq.png", ylim=[80000, 160000])
