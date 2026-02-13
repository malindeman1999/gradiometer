import numpy as np
import math
import matplotlib.pyplot as plt
from solar_wind_power_law import combined_power_law
from copy import deepcopy
import time
import numpy as np


#duh need to switch to using rfft1
class Noise:
    
    def __init__(self,period = 2**4, sample_frequency=1) -> None:
        '''
        Sets up noise generation on the time peroid, with the given sample frequence

        '''
        
        self.period=period
        self.sample_frequency = sample_frequency
        
        # Number of samples
        self.number_of_samples = int(round( self.period * self.sample_frequency))
    
        #make sure there are in even number of samples 
        if self.number_of_samples%2 !=0:
            self.number_of_samples+=1
            self.period = self.number_of_samples/self.sample_frequency
            
    
        
        # Width of each frequency bin 
        self.delta_f = self.sample_frequency/self.number_of_samples
        

        
        #the envelope is normalized such that if the integral of the 
        #envelope (in power) df =1 then the noise is multiplied by one
        #so the normalization is 1/(df*number_of_samples) (in freq domain)
        # self.envelope_norm = 1./(self.delta_f * self.number_of_samples)
        # self.sqrt_envelope_norm  = np.sqrt(self.envelope_norm)
        # Time period of each sample 
        self.delta_t = self.period/self.number_of_samples
        
        # Times
        self.times = np.linspace(0, self.period, self.number_of_samples, endpoint=False)
        
        
        
        
        self.noise = None
        self.fft_noise = None
        
        
        # Frequencies for FFT
        self.fft_freqs = np.fft.rfftfreq(self.number_of_samples, self.delta_t)
        self.number_of_frequencies = len(self.fft_freqs)
        
        self.find_fft_squared_to_power_conversion()
        self.find_white_noise_power()
        
        # Generate   noise with RMS value of 1, and its FFT is uniform 
        # noise, fft_noise = self.set_uniform_noiselike_signal()
        
        # self.check_rfft_normalization(noise, fft_noise)
         
        
        #a uniform power spectrum whose magnitud is 1/number_of_samples/delta_f integrates
        #over number_of_samples*deltaf to 1 
        #this corresponds to white noise for which the average of the squares (variarance) is 1
        self.unit_PSD =1./self.number_of_frequencies/self.delta_f*np.ones(self.number_of_frequencies)
        
        #the correcponding noise spectrum is as follows. This takes care of 
        # the fact that in rfft DC and nyquist have a power that is half as big as other freqs
        self.unit_fft_noise_squared = self.unit_PSD/self.fft2_to_power
         
         
        
        # #relate the physics unit PSD to the FFT units
        # self.PSD_to_fft2 = unit_fft_noise_squared/self.unit_PSD
        
        # PSD1 = self.compute_PSD(fft_noise)
        # variance = self.integrate_PSD(PSD1) 
        
        if  not math.isclose(self.delta_t, 1/self.sample_frequency):
            raise Exception("sample rate error")
              
        # if not math.isclose(variance, 1.):
        #     raise Exception('Error in PSD normalization')
        
        self.set_PSD_envelope()
        
        
        
    def integrate_PSD(self, PSD):
       return np.sum(PSD)*self.delta_f
   
     
    
    def noise_of_PSD(self):
        # Generate white noise with RMS value of 1
        noise = np.random.randn(self.number_of_samples)

        # FFT of noise shaped by the envelope 
        fft_noise = np.fft.rfft(noise) *self.sqrt_PSD_envelope / self.sqrt_white_noise_PSD 
        
        #convert to time domain
        noise = np.fft.irfft(fft_noise)  
        
        return noise, fft_noise      

    def compute_PSD(self,  noise):
        '''
        computes the PSD in physics units such that the integral over frequency adds up to the variance
        
        the normalization is calcuated by first running with PSD_norm =1, and these scaling norm to give correct result
        '''
         
        power = np.abs(noise)**2 * self.fft2_to_power
        
        
        return power

     
    
    def set_PSD_envelope(self,PSD_envelope = None):
        '''
        Sets the PSD to be uniform over frequency to integrate to a variance of 1 
        '''
        if PSD_envelope is None:
            PSD_envelope =  self.white_noise_PSD
            
      
        
        if len(PSD_envelope) != self.number_of_frequencies:
            raise Exception('Envelope array is not the correct length')
        
        self.PSD_envelope  = PSD_envelope
        self.sqrt_PSD_envelope = np.sqrt(self.PSD_envelope)
        
        
        
        
        
        
        


        
    def plot_sweeps_PSD(self, count = 100):
        average_fft_noise_squared = 0.
        average_mean_noise_squared = 0.
        
        t0 = time.time()
        t1= t0
        for i in range(count):
            noise, fft_noise = self.noise_of_PSD()
            
            average_mean_noise_squared += self.find_time_variance(noise)
            average_fft_noise_squared += np.abs(fft_noise)**2
            
            t2 = time.time()
            dt = t2-t1
            if dt >2:
                duration = (t2-t0)/60
                print(f'running {duration} minutes of {duration*count/i} minutes')
                t1=t2
        average_mean_noise_squared /= count
        average_fft_noise_squared /= count
        average_PSD = average_fft_noise_squared * self.fft2_to_PSD
        
        noise_squared_env = self.integrate_PSD(self.PSD_envelope)
        noise_squared_ave_PSD = self.integrate_PSD(average_PSD)
        print(f'ave noise squared   envelope: {noise_squared_env}  ave PSD: {noise_squared_ave_PSD} ave over time sweeps {average_mean_noise_squared}')
        
        #compute noise in nT rms
        Brms_nT =np.sqrt(noise_squared_env)
        
        Brms_fT = Brms_nT*1e6
        
        print(f"RMS magnetic field noise: {Brms_fT} fT ")
        
        
        # Filtering data to include only frequencies > 0  (Gradiometer completely removes zero freq anyway)
        mask = self.fft_freqs >  0
        filtered_freq = self.fft_freqs[mask]
        filtered_average_PSD = average_PSD[mask]
        filtered_envelope = self.PSD_envelope[mask]

        # Plotting on a log-log scale with the filtered data
        plt.figure(figsize=(10, 6))
        plt.loglog (filtered_freq, filtered_average_PSD, label='Average PSD', linewidth = 3)
        plt.loglog (filtered_freq, filtered_envelope, label='Envelope PSD')

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD dB$^2$/Hz (nT$^2$/Hz)')
        plt.title('Semi-Log Plot of Average PSD   vs Frequency (f > 0 Hz)')
        plt.legend()
        
        plt.show()
        
        
        
        
        
        
        
        
    def find_time_variance(self, noise): 
        '''
        takes noise over time, computes Mean Squared noise
        '''
        return np.mean( noise**2 )

    def set_solar_wind_envelope(self):
        '''
        returns power spectrum dB in nT**2 /Hz
        from Daniele Telloni, Frequency Transition From Weak to Strong Turbulence in the Solar Wind
        
        Since the PSD blows up at low frequency it makes sense to limit the power law to a minum freqency
        Really the 0 frequency bin is the PSD integrated over frequency up to the end of the fft bin (half way to the +/- lowest freq)
        The exact integral depends on the power law,
        but for now, lets assume is corresponds to 1/2 the lowest frequency
        
        really in measurements there is some power in the DC component so we shoudl not set it to zero
        I could go back and put a more precise number here, but I think the DC bin does not have the majority of the power
        '''
        self.minimum_frequency_ceiling = self.delta_f/2.
        f=deepcopy(self.fft_freqs)
        mask = f  < self.minimum_frequency_ceiling
        f[mask] = self.minimum_frequency_ceiling
        alpha1, alpha2, alpha3 = -1.46, -1.64, -2.57
        C1, C2, C3 = 4.7325896150147475, 1.7517399914430805, 0.9490611104022462 #interpreted from fig 3 
        envelope = combined_power_law(abs(f), alpha1, C1, alpha2, C2, alpha3, C3)
        
        self.set_PSD_envelope(envelope)
        
        return envelope
    
    
    def add_SQUID_noise_envelope(self):
        '''
        This function adds the SQUID PSD envelope to the envelope for other noise (such as solar wind).
        It should be applied after the gradiometer low pass filter is applied
        '''
        from squid_noise import SQUID_PSD
        f=deepcopy(self.fft_freqs)
        mask = f  < self.minimum_frequency_ceiling
        f[mask] = self.minimum_frequency_ceiling
        
        SQUID_envelope = SQUID_PSD(f)
        
        envelope =self.PSD_envelope + SQUID_envelope
        
        self.set_PSD_envelope(envelope)
        
        return  
         
    def find_fft_squared_to_power_conversion(self):
        self.fft2_to_power = np.ones(self.number_of_frequencies)*2/self.number_of_samples  
        self.fft2_to_power[0] = 1/self.number_of_samples
        self.fft2_to_power[-1] = (1+self.number_of_samples%2)/self.number_of_samples
        
        #the PSD is integrated over df so it is is different than the power
        #which is just summed
        self.fft2_to_PSD = self.fft2_to_power/self.sample_frequency
        
    def find_white_noise_power(self):
        #the what noise as equal amplitude in all freqs, but that means
        #that DC and nyquist have half the actual power  
        
        #for variance of 1 they power should average 1
        
        white_noise_squared = np.ones(self.number_of_frequencies)
        
        white_noise_power = white_noise_squared* self.fft2_to_power  
        norm = np.sum(white_noise_power)
        
        #the PSD is integrated with respect to df
        self.white_noise_PSD = white_noise_power/self.delta_f
        
        #compute the sqrt for use lateter
        self.sqrt_white_noise_PSD = np.sqrt(self.white_noise_PSD)

    def check_rfft_normalization(self, signal,fft_signal):
        # Compute the FFT
        fft_signal = np.fft.rfft(signal)
        
        
        power=np.abs(fft_signal)**2*self.fft2_to_power
        
        # Compute the sum of squares in the frequency domain
        # Note: We multiply by 2 for the positive frequency components (except for the DC and Nyquist components, if present)
        
        # freq_sum_squares = 0.5*(power[0]  +   power[-1]  * (1 + n%2))
        freq_sum_squares  =   np.sum(power)   

        # # Normalize by the number of samples
        # freq_sum_squares /= n/2.
        
        # Compute the sum of squares in the time domain
        time_sum_squares = np.sum(np.abs(signal)**2)

        # Check if they are approximately equal
        if not np.isclose(time_sum_squares, freq_sum_squares):
            raise Exception("normalization error")

    def apply_low_pass_filter(self, frequency=1, poles=1):
        filter =  (1+(self.fft_freqs/frequency)**2)**(-poles)
        envelope = self.PSD_envelope*filter
        self.set_PSD_envelope(envelope)
        
    def delay_signal(self, fft_signal, delay):
        if delay > self.period:
            raise Exception("gradiometer delay longer than peroid. Would wrap to low delay")
    
        # Phase shift
        phase_shift = np.exp(-1j * 2 * np.pi * self.fft_freqs * delay)
        
        
        return phase_shift*fft_signal 
    
    def apply_gradiometer(self, delay):
        '''
        the gradiometer measures the difference of the signal and a delayed version of it
         assuming the frozen-in flow model
         
        the delay is depence on the displament vector dotted with  the direction of flow
        
        '''
        #signal is 
        signal_multiplier =np.ones(self.number_of_frequencies)
        gradiometer_signal_multiplier = signal_multiplier - self.delay_signal(signal_multiplier,delay) 
         
        #since the PSD is mulitplied by the random phases of the white noise , we 
        #don't need to preserve the phase information after doing the substraction
        gradiometer_PSD_multiplier = np.abs( gradiometer_signal_multiplier )**2
        self.set_PSD_envelope(self.PSD_envelope*gradiometer_PSD_multiplier)
        
    def plot_noise_samples(self, quantity=3):
        plt.figure(figsize=(10, 6))

        for _ in range(quantity):
            noise, _ = self.noise_of_PSD()
            plt.plot(self.times, noise, label='Noise Sample')

        plt.xlabel('Time')
        plt.ylabel('Magnetic Noise (nT)')
        plt.title('Magnetic Noise Samples vs. Time')
        plt.legend()
        plt.show() 
        
    
    