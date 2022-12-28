import matplotlib.pyplot as plt
import memtorch

def visualize(reference_memristor,reference_memristor_params):
    memristor = reference_memristor(**reference_memristor_params)
    voltage_signal, current_signal = memristor.plot_hysteresis_loop(return_result=True)
    plt.figure()
    plt.title("Hysteresis Loop")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.plot(voltage_signal, current_signal)
    plt.savefig('Hysteresis_Loop.png')
    plt.close()
    
    voltage_signal, current_signal = memristor.plot_bipolar_switching_behaviour(return_result=True)
    plt.figure()
    plt.title("Bipolar Switching Behaviour (DC)")
    plt.xlabel("Voltage (V)")
    plt.ylabel("|Current (A)|")
    plt.plot(voltage_signal, abs(current_signal))
    plt.savefig('Bipolar_Switching_Behaviour.png')
    
if __name__=='__main__':
    reference_memristor = memtorch.bh.memristor.Stanford_PKU
    reference_memristor_params = {'time_series_resolution': 1e-10}
    visualize(reference_memristor,reference_memristor_params)