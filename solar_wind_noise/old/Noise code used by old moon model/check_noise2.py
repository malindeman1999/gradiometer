from noise_functions import Noise
import numpy as np

magnetosonic_velocity = 400e3 #m/s  400 km/S
gradiometer_length = 10   
N = Noise(period=2**16, sample_frequency=2**3)
delay = gradiometer_length/magnetosonic_velocity
print(f'delay {delay}s , samples {delay/N.delta_t}')
print('starting')
  
# N = Noise(period=2**10, sample_frequency=10)  


print(N.number_of_frequencies)
# envelope = np.ones(N.number_of_frequencies)*2      
# envelope = N.set_solar_wind_envelope()
# envelope = N.unit_PSD

# envelope[0:1] = 0
# N.set_PSD_envelope(    )
N.set_solar_wind_envelope()

# N.apply_gradiometer(delay)
# N.apply_low_pass_filter(frequency=1/4, poles = 2)

N.plot_sweeps_PSD(100)
print("Done")