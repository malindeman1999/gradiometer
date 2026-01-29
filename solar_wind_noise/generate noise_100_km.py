import numpy as np
import matplotlib.pyplot as plt
from solar_wind_power_law import combined_power_law

import numpy as np


def uniform_PSD(f,df):
        V=7.
        return V*np.ones(len(f))/len(f)/df

def uniform_PSD(f,df):
        V=10000.
        return V*np.ones(len(f))/len(f)/df

def solar_wind_function(f,df):
    #from Daniele Telloni, Frequency Transition From Weak to Strong Turbulence in the Solar Wind
    alpha1, alpha2, alpha3 = -1.46, -1.64, -2.57
    C1, C2, C3 = 4.7325896150147475, 1.7517399914430805, 0.9490611104022462 #interpreted from fig 3 
    return combined_power_law(f, alpha1, C1, alpha2, C2, alpha3, C3)

def generate_white_noise(T, fs):
    """Generate white noise with RMS value of 1.
    
    Parameters:
    - T: Time duration in seconds.
    - fs: Sampling frequency in Hz.
    
    Returns:
    - noise: White noise time series with RMS value of 1.
    - times: Array of times corresponding to noise samples.
    - delta_t: Time period of each sample.
    - fft_noise: FFT of the noise.
    - fft_freqs: Frequencies corresponding to FFT values.
    - non_neg_freqs: Frequencies that are non-negative.
    - delta_f: Width of each frequency bin.
    """
    
    # Number of samples
    N = int(T * fs)
    
    # Generate white noise with RMS value of 1
    noise = np.random.randn(N)
    
    # Times
    times = np.linspace(0, T, N, endpoint=False)
    
    # Time period of each sample
    delta_t = T/N
    
    # FFT of noise
    fft_noise = np.fft.fft(noise)
    
    # Frequencies for FFT
    fft_freqs = np.fft.fftfreq(N, 1/fs)
    
    # Only non-negative frequencies
    non_neg_freqs = fft_freqs[fft_freqs >= 0]
    
    
    #check that GPT got his right:
    # Width of each frequency bin 
    delta_f = fs/N
    
    return noise, times, delta_t, fft_noise, fft_freqs, non_neg_freqs, delta_f


def replace_zero_frequency(freqs):
    # Find the smallest positive frequency
    min_positive = min([f for f in freqs if f > 0])
    
    # Replace the 0 frequency with the smallest positive frequency
    freqs_modified = np.where(freqs == 0, min_positive, freqs)
    
    return freqs_modified

def validate_realness(signal, threshold=30):
    """
    Check if the imaginary part of a signal has significantly less power than the real part.
    
    Parameters:
    - signal: The complex-valued signal.
    - threshold: The minimum acceptable signal-to-noise ratio (SNR) in dB. Default is 30 dB.
    
    Returns:
    - A boolean indicating whether the imaginary power is negligible.
    """
    # Compute power of the real and imaginary parts
    real_power = np.sum(np.real(signal)**2)
    imag_power = np.sum(np.imag(signal)**2)
    
    # Compute the SNR in dB
    snr = 10 * np.log10(real_power / (imag_power + 1e-10))  # added a small value to avoid division by zero
    
    if snr < threshold:
        raise Exception(f"SNR is below the threshold! SNR: {snr:.2f} dB")
    
    return np.real(signal)

# compute total PSD power for double checking. Note that   DC and Nyquist are treated properly because
# the power in those comes from the fft.
def integrate_PSD(PSD,df):
    variance = np.sum(PSD)*df
    print(f"length of PSD {len(PSD)} ")
    return variance

def time_domain_variance(time_sample):
    variance = np.var(time_sample)
    return variance


def compute_noise_sample(T_sample=25000, fs_sample = 4):
    # Demonstrate with an example   #25000 s, 4 Hz sampling
   
   #white noise with variance =1
    noise_sample, times_sample, dt, fft_sample, freqs_sample, non_neg_freqs, df = generate_white_noise(T_sample, fs_sample)


    #power law blows up at low frequency
    freqs_sample_modified = replace_zero_frequency(freqs_sample)
    non_neg_freqs_modified = replace_zero_frequency(non_neg_freqs)


    number_of_frequency_bins=len(non_neg_freqs_modified)
 

    
    normalization=2*df*number_of_frequency_bins
    PSD  = func(non_neg_freqs_modified, df)
    multiplier_squared = func(freqs_sample_modified, df)*normalization
    multiplier= multiplier_squared**.5

    solar_wind_fft=fft_sample*multiplier
    time_signal = np.fft.ifft(solar_wind_fft)

    solar_wind_signal= validate_realness(time_signal)
    
    return  solar_wind_signal, times_sample, solar_wind_fft, freqs_sample, normalization, PSD,df









# func = uniform_PSD
func = solar_wind_function


magnetosonic_velocity = 400e3 #m/s  400 km/S
Europa_diameter = 3100e3 # m       #3100 kilometers
Europa_transit_time = Europa_diameter / magnetosonic_velocity
Europa_transit_frequency = 10. / Europa_transit_time

gradiometer_length=100e3  # m

sample_period=gradiometer_length/magnetosonic_velocity
sample_frequency=1/sample_period*2**10

print(f"Europa transit time: {Europa_transit_time}  Transit frequency: {Europa_transit_frequency}")


a=0.
number_of_samples=30
for i in range(number_of_samples):
    noise_t, t, noise_f, f, normalization, PSD, df= compute_noise_sample (T_sample=sample_period, fs_sample= sample_frequency)
    
    
    a = a+np.abs(noise_f)**2



length_of_PSD_integral=df*len(PSD)

fft_norm=1/len(t)
PSD_ave=  a*fft_norm**2/df/number_of_samples

nf=np.fft.fft(noise_t)*fft_norm
# print(f"ratio  {np.sum(PSD_ave)/np.sum(np.abs(nf)**2)}")

print(f"PSE_ave integral;:{np.sum(PSD_ave)*df  }   time integral {np.mean(np.abs(noise_t)**2)}")

# print(f"FREQ integra;:{np.sum(np.abs(nf)**2)}   time integral {np.mean(np.abs(noise_t)**2)}")

# print(f"noise **2 averaged{ (np.mean(noise_t**2)-np.mean(noise_t)**2) }")
# print(f"mean squared: { np.mean(noise_t**2)  }  squared mean: { np.mean(noise_t)**2  }  ")
 
PSD_theory  = func(f, df)



position=t*magnetosonic_velocity    # position in meters

distance_scale=max(position)/magnetosonic_velocity

print(f"Distance scale :{distance_scale} Europa diameters,  {max(position )} meters")

vt = time_domain_variance(noise_t)
vPSD = integrate_PSD(PSD,df)   # need to resolve why these are only the same for white noise, but differ for non white.

rms_t = np.sqrt(vt)
rms_PSD = np.sqrt(vPSD)

print(f"variance {vt} nT^2  PSD estimated variance: {vPSD} nT^2")
print(f"rms {rms_t*1e6} fT   PSD variance: {rms_PSD} nT")
print('done')

# Plot the time domain sample (first 10 ms) and its FFT
fig, ax = plt.subplots(3, 1, figsize=(10, 8))

# Frequency domain
ax[0].loglog(f , PSD_ave)
ax[0].loglog(f , PSD_theory,c='red')
ax[0].set_title(" Noise Frequency Spectrum")
ax[0].set_xlabel("Frequency [Hz]")
ax[0].set_ylabel("Amplitude")



# Time domainS
ax[1].plot(t , noise_t )
ax[1].set_title("White Noise Time Domain")
ax[1].set_xlabel("Time [s]")
ax[1].set_ylabel("Amplitude nT")



# position domain
ax[2].plot(position , noise_t )
ax[2].set_title("White Noise Space Domain")
ax[2].set_xlabel("position (m)")
ax[2].set_ylabel("Amplitude (nT)")

 
plt.tight_layout()
plt.show()

