import numpy as np
import matplotlib.pyplot as plt
from solar_wind_power_law import combined_power_law

# Define constants
fs = 200e3  # Sampling frequency in Hz
T = 10  # Duration in seconds
N = int(T * fs)  # Total number of samples
freqs = np.fft.fftfreq(N, 1/fs)

def generate_noise_with_psd(frequencies, psd_function, N, fs):
    # Step 1: Generate complex white noise in frequency domain
    white_noise_freq = (np.random.randn(N) + 1j * np.random.randn(N))
    
    # Step 2: Multiply amplitude by square root of desired PSD
    desired_amplitude = np.sqrt(psd_function(np.abs(frequencies)))
    noise_freq = white_noise_freq * desired_amplitude
    
    # Step 3: Inverse FFT to get time-domain signal
    noise_time = np.fft.ifft(noise_freq).real
    
    return noise_time

# Define the combined PSD function
def combined_psd_function(frequencies):
    alpha1, alpha2, alpha3 = -1.46, -1.64, -2.57
    C1, C2, C3 = 4.7325896150147475, 1.7517399914430805, 0.9490611104022462
    return  combined_power_law(frequencies, alpha1, C1, alpha2, C2, alpha3, C3)

# Generate noise with desired PSD
noise = generate_noise_with_psd(freqs, combined_psd_function, N, fs)

# Plot a short segment of the noise
plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, 0.01, int(0.01*fs)), noise[:int(0.01*fs)])
plt.title("Generated Magnetic Noise (First 10 ms)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()
