import numpy as np

def check_rfft_normalization(signal):
    # Compute the FFT
    fft_result = np.fft.rfft(signal)
    
    # Compute the sum of squares in the frequency domain
    # Note: We multiply by 2 for the positive frequency components (except for the DC and Nyquist components, if present)
    n = len(signal)
    freq_sum_squares = np.abs(fft_result[0])**2 + np.abs(fft_result[-1])**2 * (1 + n%2)
    freq_sum_squares += 2 * np.sum(np.abs(fft_result[1:n//2])**2) if n > 2 else 0

    # Normalize by the number of samples
    freq_sum_squares /= n
    
    # Compute the sum of squares in the time domain
    time_sum_squares = np.sum(np.abs(signal)**2)

    # Check if they are approximately equal
    return np.isclose(time_sum_squares, freq_sum_squares)

# Example usage
signal = np.random.random(1024)  # A random signal
print("Normalization check:", check_rfft_normalization(signal))
