from noise_functions import Noise
import numpy as np

#Call to get freq range and normalization
#needs to agree with call below

period= 2**10
fs = 30

noise = Noise()

noise_sample, times_sample, dt, fft_noise, fft_freqs, non_neg_freqs, df, freq_norm = noise.generate_white_noise(T=period, fs = fs)

uniform_PSD_envelope


#assume that the PSD integrated over the DC is roughly the value at df/2
# I could calculate a more accurate number by integrating the power law
# PSD_envelope = solar_wind_function(fft_freqs, df, minimum_frequency_ceiling=df/2 )
PSD_envelope = uniform_PSD_envelope(fft_freqs, df)

variance_f_envelope = np.sum(PSD_envelope)*df


noise_envelope = np.sqrt(PSD_envelope)




variance_f = []
variance_t = []
variance_raw = []
for i in range(1000):
    noise_sample, times_sample, dt, fft_noise, fft_freqs, non_neg_freqs, df, freq_norm = generate_white_noise(T=period, fs = fs)
    variance_raw.append(np.mean(noise_sample**2))
    
    fft_solar_noise = fft_noise*noise_envelope
    #normalize power spectrum such that it integrates to the variance in the time domain
    fft_solar_power = np.abs(fft_solar_noise)**2*freq_norm
    variance_f.append(  np.sum(fft_solar_power)*df )

    solar_noise = np.fft.ifft(fft_solar_noise)
    solar_power = np.abs(solar_noise)**2
    variance_t.append( np.mean(solar_power))


print(f'Variance uniform: {np.mean(variance_raw)}  envelope {variance_f_envelope} frequency {np.mean(variance_f)} time {np.mean(variance_t)}')
print(f"normalization: {freq_norm}")