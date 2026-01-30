from noise_functions import Noise
import numpy as np

magnetosonic_velocity = 400e3 #m/s  400 km/S
gradiometer_length = 10   
N = Noise(period=2**19, sample_frequency=1.)
delay = gradiometer_length/magnetosonic_velocity
print(f'delay {delay}s , samples {delay/N.delta_t}')
print('starting')
  
# N = Noise(period=2**10, sample_frequency=10)  


print(N.number_of_frequencies)

N.set_solar_wind_envelope()

# N.apply_gradiometer(delay)
N.apply_low_pass_filter(frequency=1/4, poles = 2)

N.plot_sweeps_PSD(100)
N.plot_noise_samples()
print("Done")