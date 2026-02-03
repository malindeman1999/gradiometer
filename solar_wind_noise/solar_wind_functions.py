import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from solar_wind_power_law import combined_power_law


@dataclass(frozen=True)
class EuropaProperties:
    """Physical constants for Europa transit calculations."""

    magnetosonic_velocity: float = 400e3
    diameter: float = 3100e3

    @property
    def transit_time(self):
        """Transit time across Europa in seconds."""
        return self.diameter / self.magnetosonic_velocity


def uniform_PSD(f, df):
    """Return a flat (white) power spectral density for the given frequencies.

    Parameters:
    - f: Array of frequencies in Hz.
    - df: Frequency bin width in Hz.
    """
    V = 10000.0
    return V * np.ones(len(f)) / len(f) / df


def solar_wind_function(f, df):
    """
    Returns power spectrum dB in nT**2 / Hz.
    From Daniele Telloni, Frequency Transition From Weak to Strong Turbulence in the Solar Wind.
    """
    alpha1, alpha2, alpha3 = -1.46, -1.64, -2.57
    C1, C2, C3 = 4.7325896150147475, 1.7517399914430805, 0.9490611104022462
    return combined_power_law(f, alpha1, C1, alpha2, C2, alpha3, C3)


def gradiometer_solar_wind_function(
    f,
    df,
    gradiometer_length,
    angle_to_solar_wind=0.0,
    magnetosonic_velocity=EuropaProperties().magnetosonic_velocity,
    number_of_points=5,
):
    """Return the solar-wind PSD as seen by a multipoint gradiometer differencer.

    Parameters:
    - f: Frequency array in Hz.
    - df: Frequency bin width in Hz (included for interface compatibility).
    - gradiometer_length: Sensor separation in meters.
    - angle_to_solar_wind: Angle (radians) between gradiometer baseline and flow.
      A value of 0 means aligned with the solar-wind direction.
    - magnetosonic_velocity: Advection speed in m/s used for frozen-in mapping.
    - number_of_points: Number of points in the gradiometer model.
      Special cases:
        * 1: single-point field measurement (no differencing).
        * 2: two-point endpoint difference across the full baseline.
      For values >= 3, must be an odd integer and uses a central-difference stencil.
    """
    transfer_power = gradiometer_transfer_power(
        f=f,
        gradiometer_length=gradiometer_length,
        angle_to_solar_wind=angle_to_solar_wind,
        magnetosonic_velocity=magnetosonic_velocity,
        number_of_points=number_of_points,
    )
    return transfer_power * solar_wind_function(f, df)


def gradiometer_transfer_power(
    f,
    gradiometer_length,
    angle_to_solar_wind=0.0,
    magnetosonic_velocity=EuropaProperties().magnetosonic_velocity,
    number_of_points=5,
):
    """Return peak-normalized gradiometer transfer power |H(f)|^2 (dimensionless)."""
    if gradiometer_length <= 0:
        raise ValueError("gradiometer_length must be positive.")
    if magnetosonic_velocity <= 0:
        raise ValueError("magnetosonic_velocity must be positive.")
    if number_of_points < 1 or int(number_of_points) != number_of_points:
        raise ValueError("number_of_points must be an integer >= 1.")

    d_parallel = gradiometer_length * np.cos(angle_to_solar_wind)
    angular_frequency = 2 * np.pi * f

    if number_of_points == 1:
        return np.ones_like(f, dtype=float)

    if number_of_points == 2:
        delay = d_parallel / magnetosonic_velocity
        transfer = np.exp(1j * angular_frequency * delay) - 1.0
        transfer_power = np.abs(transfer) ** 2
        peak = np.max(transfer_power)
        return transfer_power / peak if peak > 0 else transfer_power

    offsets, coeffs = central_difference_coefficients(number_of_points)
    delay_step = d_parallel / magnetosonic_velocity / (number_of_points - 1)
    transfer = np.zeros_like(f, dtype=np.complex128)
    for offset, coeff in zip(offsets, coeffs):
        transfer += coeff * np.exp(1j * angular_frequency * offset * delay_step)
    transfer_power = np.abs(transfer) ** 2
    peak = np.max(transfer_power)
    return transfer_power / peak if peak > 0 else transfer_power


def central_difference_coefficients(number_of_points):
    """Return offsets and coefficients for a central finite-difference gradiometer.

    The returned coefficients estimate a first derivative and are scaled by the
    total baseline so that they reduce to a simple two-endpoint difference for
    a 3-point stencil.
    """
    if number_of_points < 3 or number_of_points % 2 == 0:
        raise ValueError(
            "For central-difference coefficients, number_of_points must be an odd integer >= 3."
        )

    half_width = number_of_points // 2
    offsets = np.arange(-half_width, half_width + 1, dtype=int)

    # Solve moment-matching equations for first-derivative weights at x=0, h=1.
    system = np.array([offsets.astype(float) ** p for p in range(number_of_points)])
    rhs = np.zeros(number_of_points, dtype=float)
    rhs[1] = 1.0
    derivative_weights = np.linalg.solve(system, rhs)

    # Scale by the full baseline length so output is "difference-like" in nT.
    coefficients = (number_of_points - 1) * derivative_weights
    return offsets, coefficients


def generate_white_noise(T, fs, rng=None):
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
    if rng is None:
        rng = np.random.default_rng()

    # Number of samples
    N = int(T * fs)

    # Generate white noise with RMS value of 1
    noise = rng.standard_normal(N)

    # Times
    times = np.linspace(0, T, N, endpoint=False)

    # Time period of each sample
    delta_t = T / N

    # FFT of noise
    fft_noise = np.fft.fft(noise)

    # Frequencies for FFT
    fft_freqs = np.fft.fftfreq(N, 1 / fs)

    # Only non-negative frequencies
    non_neg_freqs = fft_freqs[fft_freqs >= 0]

    # Width of each frequency bin
    delta_f = fs / N

    return noise, times, delta_t, fft_noise, fft_freqs, non_neg_freqs, delta_f


def replace_zero_frequency(freqs):
    """Replace zero entries with the smallest positive frequency.

    Parameters:
    - freqs: Array of frequency values.
    """
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
    real_power = np.sum(np.real(signal) ** 2)
    imag_power = np.sum(np.imag(signal) ** 2)

    # Compute the SNR in dB
    snr = 10 * np.log10(real_power / (imag_power + 1e-10))

    if snr < threshold:
        raise Exception(f"SNR is below the threshold! SNR: {snr:.2f} dB")

    return np.real(signal)


# compute total PSD power for double checking. Note that DC and Nyquist are treated properly because
# the power in those comes from the fft.
def integrate_PSD(PSD, df):
    """Integrate a one-sided PSD to estimate variance.

    Parameters:
    - PSD: One-sided power spectral density values.
    - df: Frequency bin width in Hz.
    """
    variance = np.sum(PSD) * df
    print(f"length of PSD {len(PSD)} ")
    return variance


def time_domain_variance(time_sample):
    """Compute variance of a time-domain signal.

    Parameters:
    - time_sample: Array of time-domain samples.
    """
    variance = np.var(time_sample)
    return variance


def compute_noise_sample(T_sample, fs_sample, func, rng=None):
    """Generate a noise realization with a target PSD.

    Parameters:
    - T_sample: Time duration in seconds.
    - fs_sample: Sampling frequency in Hz.
    - func: PSD function taking (freqs, df) and returning PSD values.
    """
    # white noise with variance = 1
    noise_sample, times_sample, dt, fft_sample, freqs_sample, non_neg_freqs, df = generate_white_noise(
        T_sample, fs_sample, rng=rng
    )

    # power law blows up at low frequency
    freqs_sample_modified = replace_zero_frequency(freqs_sample)
    non_neg_freqs_modified = replace_zero_frequency(non_neg_freqs)

    number_of_frequency_bins = len(non_neg_freqs_modified)

    normalization = 2 * df * number_of_frequency_bins
    PSD = func(non_neg_freqs_modified, df)
    multiplier_squared = func(freqs_sample_modified, df) * normalization
    multiplier = multiplier_squared ** 0.5

    solar_wind_fft = fft_sample * multiplier
    time_signal = np.fft.ifft(solar_wind_fft)

    solar_wind_signal = validate_realness(time_signal)

    return solar_wind_signal, times_sample, solar_wind_fft, freqs_sample, normalization, PSD, df


def compute_average_psd(T_sample, fs_sample, number_of_samples, func, seed=None):
    """Average PSD across multiple noise realizations.

    Parameters:
    - T_sample: Time duration in seconds.
    - fs_sample: Sampling frequency in Hz.
    - number_of_samples: Number of realizations to average.
    - func: PSD function taking (freqs, df) and returning PSD values.
    - seed: Optional random seed for reproducible realizations.
    """
    rng = np.random.default_rng(seed)

    a = 0.0
    diff_power = 0.0
    for _ in range(number_of_samples):
        noise_t, t, noise_f, f, normalization, PSD, df = compute_noise_sample(
            T_sample=T_sample, fs_sample=fs_sample, func=func, rng=rng
        )
        a = a + np.abs(noise_f) ** 2
        diff = noise_t[-1] - noise_t[0]
        diff_power += diff * diff

    fft_norm = 1 / len(t)
    PSD_ave = a * fft_norm**2 / df / number_of_samples
    diff_rms = np.sqrt(diff_power / number_of_samples)
    return noise_t, t, noise_f, f, normalization, PSD, df, PSD_ave, diff_rms


def plot_noise_results(
    f,
    PSD_ave,
    PSD_theory,
    t,
    noise_t,
    position,
    gradiometer_psd_ave=None,
    gradiometer_psd_theory=None,
    gradiometer_transfer_power=None,
    gradiometer_transfer_frequency=None,
    gradiometer_noise_t=None,
    gradiometer_t=None,
    gradiometer_points=None,
):
    """Plot PSD, time-domain signal, and position-domain signal.

    Parameters:
    - f: Frequency array in Hz.
    - PSD_ave: Estimated PSD from averaged FFTs.
    - PSD_theory: Target/theoretical PSD curve.
    - t: Time array in seconds.
    - noise_t: Time-domain noise samples.
    - position: Position array in meters.
    - gradiometer_psd_ave: Optional averaged gradiometer PSD from realizations.
    - gradiometer_psd_theory: Optional gradiometer PSD curve for differenced signal.
    - gradiometer_transfer_power: Optional transfer power |H(f)|^2 for gradiometer.
    - gradiometer_transfer_frequency: Optional frequency array for transfer-power plot.
    - gradiometer_noise_t: Optional time-domain gradiometer noise samples.
    - gradiometer_t: Optional time array for gradiometer noise samples.
    - gradiometer_points: Optional integer count used in gradiometer plot titles.
    """
    amplitude_scale = 1e3  # convert nT -> pT for plotting
    psd_scale = amplitude_scale**2  # convert nT^2/Hz -> pT^2/Hz

    num_plots = 3
    if gradiometer_psd_theory is not None:
        num_plots += 1
    if gradiometer_transfer_power is not None:
        num_plots += 1
    if gradiometer_noise_t is not None and gradiometer_t is not None:
        num_plots += 1

    ncols = 3
    nrows = int(np.ceil(num_plots / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.8 * nrows))
    ax = np.atleast_1d(ax).ravel()

    # Frequency-domain plots should use positive frequencies only.
    positive_freq_mask = f > 0
    f_plot = f[positive_freq_mask]
    psd_ave_plot = PSD_ave[positive_freq_mask] * psd_scale
    psd_theory_plot = PSD_theory[positive_freq_mask] * psd_scale

    # Frequency domain
    ax[0].loglog(f_plot, psd_ave_plot, linewidth=2.2)
    ax[0].loglog(f_plot, psd_theory_plot, c="red")
    ax[0].set_title("Noise Frequency Spectrum")
    ax[0].set_xlabel("Frequency [Hz]")
    ax[0].set_ylabel("PSD [pT^2/Hz]")

    next_plot = 1
    if gradiometer_transfer_power is not None:
        transfer_freq = f if gradiometer_transfer_frequency is None else gradiometer_transfer_frequency
        transfer_mask = transfer_freq > 0
        transfer_freq_plot = transfer_freq[transfer_mask]
        transfer_power_plot = gradiometer_transfer_power[transfer_mask]
        ax[next_plot].semilogx(transfer_freq_plot, transfer_power_plot, linewidth=2.2)

        ax[next_plot].set_title("Gradiometer Transfer Power")
        ax[next_plot].set_xlabel("Frequency [Hz]")
        ax[next_plot].set_ylabel("Transfer Power [1]")
        next_plot += 1

    if gradiometer_psd_theory is not None:
        if gradiometer_psd_ave is not None:
            gradiometer_psd_ave_plot = gradiometer_psd_ave[positive_freq_mask] * psd_scale
            ax[next_plot].loglog(f_plot, gradiometer_psd_ave_plot, linewidth=2.2)
        gradiometer_psd_plot = gradiometer_psd_theory[positive_freq_mask] * psd_scale
        ax[next_plot].loglog(f_plot, gradiometer_psd_plot, c="red")
        grad_title = "Noise Frequency Spectrum of the solar wind measured by the gradiometer"
        if gradiometer_points is not None:
            grad_title += f" ({gradiometer_points} points)"
        ax[next_plot].set_title(grad_title)
        ax[next_plot].set_xlabel("Frequency [Hz]")
        ax[next_plot].set_ylabel("PSD [pT^2/Hz]")
        next_plot += 1

    # Time domain
    ax[next_plot].plot(t, noise_t * amplitude_scale)
    ax[next_plot].set_title("Solar Wind Noise Time Domain")
    ax[next_plot].set_xlabel("Time [s]")
    ax[next_plot].set_ylabel("Amplitude [pT]")
    next_plot += 1

    # Position domain
    ax[next_plot].plot(position, noise_t * amplitude_scale)
    ax[next_plot].set_title("Solar Wind Noise Space Domain")
    ax[next_plot].set_xlabel("Position [m]")
    ax[next_plot].set_ylabel("Amplitude [pT]")
    next_plot += 1

    if gradiometer_noise_t is not None and gradiometer_t is not None:
        ax[next_plot].plot(gradiometer_t, gradiometer_noise_t * amplitude_scale)
        ax[next_plot].set_title("Gradiometer Noise Time Domain (Last Realization)")
        ax[next_plot].set_xlabel("Time [s]")
        ax[next_plot].set_ylabel("Amplitude [pT]")
        next_plot += 1

    for i in range(next_plot, len(ax)):
        ax[i].set_visible(False)

    plt.tight_layout()
    plt.show()


def apply_gradiometer_difference(noise_t, fs, tau):
    """Apply a gradiometer difference in the time domain.

    Parameters:
    - noise_t: Time-domain noise samples.
    - fs: Sampling frequency in Hz.
    - tau: Time delay in seconds.
    """
    N = len(noise_t)
    delay_samples = int(round(tau * fs))
    if delay_samples <= 0:
        raise ValueError("Computed delay in samples is out of valid range.")
    if delay_samples >= N:
        delay_samples = N - 1
        print(
            "Warning: delay equals/exceeds record length; "
            "using N-1 for time-domain difference."
        )
    diff = noise_t[delay_samples:] - noise_t[:-delay_samples]
    t_diff = np.arange(delay_samples, N) / fs
    return diff, t_diff, delay_samples




def summarize_noise_results(f, df, PSD, PSD_ave, noise_t, t, magnetosonic_velocity, func, diff_rms):
    """Compute and print common noise diagnostics.

    Parameters:
    - f: Frequency array in Hz.
    - df: Frequency bin width in Hz.
    - PSD: One-sided PSD used for variance estimation.
    - PSD_ave: Estimated PSD from averaged FFTs.
    - noise_t: Time-domain noise samples.
    - t: Time array in seconds.
    - magnetosonic_velocity: Flow speed in m/s.
    - func: PSD function taking (freqs, df) and returning PSD values.
    - diff_rms: RMS of the gradiometer-endpoint difference in nT.
    """
    print(f"PSE_ave integral;:{np.sum(PSD_ave) * df  }   time integral {np.mean(np.abs(noise_t) ** 2)}")

    PSD_theory = func(f, df)
    position = t * magnetosonic_velocity
    distance_scale = max(position) / magnetosonic_velocity

    print(f"Distance scale :{distance_scale} Europa diameters,  {max(position )} meters")

    vt = time_domain_variance(noise_t)
    vPSD = integrate_PSD(PSD, df)

    rms_t = np.sqrt(vt)
    rms_PSD = np.sqrt(vPSD)

    print(f"variance {vt} nT^2  PSD estimated variance: {vPSD} nT^2")
    print(f"Actual RMS {rms_t*1e6} fT   PSD estimated RMS: {rms_PSD*1e6} fT")
    print(f"gradiometer difference RMS: {diff_rms*1e6} fT")
    print("Note: PSD estimates assume extended averaging; the actual RMS over a single transit is smaller.")
    print("done")

    return PSD_theory, position
