import memtorch
import matplotlib.pyplot as plt
import os

import numpy as np


def plot_mem_fluctuation(x_data, y_data, x_label: str, y_label: str, title: str, save_path: str, ylim=None,
                         log_scale=False, use_line=False, show=True):
    plt.figure()
    plt.title(title)
    plt.xlabel(x_label)
    if use_line:
        plt.plot(x_data, y_data)
    plt.scatter(x_data, y_data, s=15, c=np.linspace(0, 1, num=x_data.shape[0]))
    plt.ylabel(y_label)
    if not os.path.exists('figures'):
        os.mkdir('figures')
    if log_scale:
        plt.yscale("log")
    if ylim:
        plt.ylim(ylim)
    plt.savefig(os.path.join('figures', save_path))
    if show:
        plt.show()
    plt.close()


if __name__ == '__main__':
    reference_memristor = memtorch.bh.memristor.VTEAM
    reference_memristor_params = {
        'r_on': 100, 'r_off': 2500,
        'v_off': 0.5, 'v_on': -0.53,
        'k_off': 4.03e-8, 'k_on': -80,
        'x_off': 1e-8, 'x_on': 0}
    eps = 1e-12  # prevent divide by zero
    memristor = reference_memristor(time_series_resolution=2e-5,**reference_memristor_params)

    voltage, current = memristor.plot_hysteresis_loop(
        duration=2,
        voltage_signal_amplitude=1,
        voltage_signal_frequency=0.5,
        return_result=True
    )
    slices = np.abs(voltage) > eps
    voltage, current = voltage[slices], current[slices]
    R = voltage / (current + eps)
    plot_mem_fluctuation(voltage, current, "Voltage (V)", "Current (A)", "Hysteresis Loop-omega=pi",
                         "memristor-hys-loop-3-I-U.png")
    plot_mem_fluctuation(voltage, R, "Voltage (V)", "Resistance (Ohm)", "Hysteresis Loop-omega=pi",
                         "memristor-hys-loop-3-R-U.png")

    memristor = reference_memristor(time_series_resolution=1e-6, **reference_memristor_params)
    voltage, current = memristor.plot_hysteresis_loop(
        duration=0.001,
        voltage_signal_amplitude=1,
        voltage_signal_frequency=1000,
        return_result=True
    )
    slices = np.abs(voltage) > eps
    voltage, current = voltage[slices], current[slices]
    R = voltage / (current + eps)
    plot_mem_fluctuation(voltage, current, "Voltage (V)", "Current (A)", "Hysteresis Loop-omega=1000pi",
                         "memristor-hys-loop-high-I-U.png")
    plot_mem_fluctuation(voltage, R, "Voltage (V)", "Resistance (Ohm)", "Hysteresis Loop-omega=1000pi",
                         "memristor-hys-loop-high-R-U.png",ylim=[0,130])

    memristor = reference_memristor(**reference_memristor_params)
    voltage, current = memristor.plot_bipolar_switching_behaviour(
        log_scale=True,
        return_result=True
    )
    slices = np.abs(voltage) > eps
    voltage, current = voltage[slices], current[slices]
    R = voltage / (current + eps)
    plot_mem_fluctuation(voltage, np.abs(current), "Voltage (V)", "Current (A)", "Bipolar Loop",
                         "memristor-bipolar-loop-I-U.png", use_line=True, log_scale=True)
    plot_mem_fluctuation(voltage, R, "Voltage (V)", "Resistance (Ohm)", "Bipolar Loop",
                         "memristor-bipolar-loop-R-U.png", use_line=True)
