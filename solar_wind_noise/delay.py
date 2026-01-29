import numpy as np

def delay_signal(signal, sampling_rate, delay):
    n = len(signal)
    freq = np.fft.rfftfreq(n, d=1/sampling_rate)  # Frequency array

    # Fourier transform
    fft_signal = np.fft.rfft(signal)

    # Phase shift
    phase_shift = np.exp(-1j * 2 * np.pi * freq * delay)
    fft_signal_shifted = fft_signal * phase_shift

    # Inverse Fourier transform
    delayed_signal = np.fft.irfft(fft_signal_shifted, n)

    return delayed_signal

# Example usage
sampling_rate = 1000  # Example sampling rate in Hz
signal = np.random.random(16)  # Example signal
delay = 0.001*1  # Delay in seconds

delayed_signal = delay_signal(signal, sampling_rate, delay)

print(signal)
print(delayed_signal)