import numpy as np
import matplotlib.pyplot as plt

def power_law(f, alpha, C=1):
    """Power law function."""
    return C * f**alpha

def calculate_C(delta_B2, f, alpha):
    """Calculate the normalization constant C based on a known point and alpha."""
    return delta_B2 / (f**alpha)
def squid_noise(frequencies_signed ):
    """SQUID noise in fT/ root Hz to match Robin Cantor paper.
    ##assumes a power function that is 26 fT/root Hz at 0 Hz and 
    This needs to be squared to make a PSD and then converted to nT^2 / Hz.  Then add to solar wind noise PSD  after the gradiometer low pass filter is included
    
    """ 
    frequencies=np.abs(frequencies_signed)  # fft may return negative frequencies
    
    #approximate slope of noise from log log plot in Cantor paper
    
    #The noise at two frequencies
    n0 = 26   #fT/ rt Hz   
    f0 =1     #Hz
        
    n1 = 10  ##fT/ rt Hz 
    f1 = 10000 #  Hz 
    
    #The slope of the noise in log log space
    p = np.log(n1/n0)/np.log(f1)
    noise =n0*frequencies**(p)
    
    
    # Ensuring the noise does not fall below 10 fT/√Hz
    minimum_noise = 10  # fT/√Hz
    noise = np.maximum(noise, minimum_noise)
    
    return noise

def SQUID_PSD(frequencies):
    '''
    returns the noise PSD in nT^2/Hz
    '''
    #confert noise from fT to nT
    noise = squid_noise(frequencies) *1e-6
    PSD = noise**2    #nT^2 / Hz 
    
    return PSD

import matplotlib.pyplot as plt


# frequencies = np.linspace(1e-5, 20000, 500)  # Frequencies from 1e-5 Hz to 10 kHz
# PSD = SQUID_PSD(frequencies)
# plt.figure(figsize=(10, 6))
# plt.plot(frequencies, PSD, label="SQUID PSD in nT*2/Hz")
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('SQUID PSD  nT^2/ Hz)')
# plt.title('Modified User-Defined SQUID Noise vs Frequency')
# plt.xscale('log')
# plt.yscale('log')
# plt.grid(True)
# plt.legend()
# plt.show()


# noise = squid_noise(frequencies)
# plt.figure(figsize=(10, 6))
# plt.plot(frequencies, noise, label="Modified User-Defined SQUID Noise")
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Noise (fT/√Hz)')
# plt.title('Modified User-Defined SQUID Noise vs Frequency')
# plt.xscale('log')
# plt.yscale('log')
# plt.grid(True)
# plt.legend()
# plt.show()

