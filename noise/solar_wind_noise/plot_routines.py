"""Shared plotting routines for solar wind noise scripts."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

def _central_difference_coefficients(number_of_points):
    """Return offsets and coefficients for a central finite-difference stencil."""
    if number_of_points < 3 or number_of_points % 2 == 0:
        raise ValueError(
            "For central-difference coefficients, number_of_points must be an odd integer >= 3."
        )
    half_width = number_of_points // 2
    offsets = np.arange(-half_width, half_width + 1, dtype=int)
    system = np.array([offsets.astype(float) ** p for p in range(number_of_points)])
    rhs = np.zeros(number_of_points, dtype=float)
    rhs[1] = 1.0
    derivative_weights = np.linalg.solve(system, rhs)
    coefficients = (number_of_points - 1) * derivative_weights
    return offsets, coefficients


def centered_correlation(signal, sample_dt):
    """Compute normalized autocorrelation and approximate 1-sigma errors."""
    values = np.asarray(signal, dtype=float)
    center = values.size // 2
    half_window = min(center, values.size - center - 1)
    window = values[center - half_window : center + half_window + 1]
    window = window - np.mean(window)

    correlation_raw = np.correlate(window, window, mode="full")
    lags = np.arange(-(window.size - 1), window.size)
    overlap = window.size - np.abs(lags)
    correlation = correlation_raw / (np.var(window) * overlap)
    corr_stderr = 1.0 / np.sqrt(overlap)
    return lags * sample_dt, correlation, corr_stderr


def plot_centered_correlation(
    lag_noise,
    corr_noise,
    err_noise,
    lag_grad,
    corr_grad,
    err_grad,
    *,
    max_lag_seconds: float | None = None,
    title: str = "Centered Correlation Function",
):
    if max_lag_seconds is None:
        mask = lag_noise > 0
    else:
        mask = (lag_noise > 0) & (lag_noise <= max_lag_seconds)

    plt.figure(figsize=(10, 4))
    plt.errorbar(
        lag_noise[mask],
        corr_noise[mask],
        yerr=err_noise[mask],
        fmt="o",
        markersize=2.5,
        elinewidth=0.7,
        capsize=0,
        linestyle="none",
        alpha=0.8,
        label="Solar wind noise",
    )
    plt.errorbar(
        lag_grad[mask],
        corr_grad[mask],
        yerr=err_grad[mask],
        fmt="o",
        markersize=2.5,
        elinewidth=0.7,
        capsize=0,
        linestyle="none",
        alpha=0.8,
        label="Gradiometer noise",
    )
    plt.xscale("log")
    plt.title(title)
    plt.xlabel("Positive lag [s]")
    plt.ylabel("Correlation [1]")
    plt.ylim(-1.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


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
    gradiometer_length=None,
    gradiometer_output_quantity=None,
    gradiometer_amplitude_scale=None,
    gradiometer_psd_scale=None,
    gradiometer_amplitude_label="Amplitude [pT]",
    gradiometer_psd_label="PSD [pT^2/Hz]",
    gradiometer_plot_asd=False,
    gradiometer_rms_band=None,
    sensor_rms_band=None,
):
    """Plot PSD, time-domain signal, and position-domain signal."""
    amplitude_scale = 1e3  # convert nT -> pT for plotting
    psd_scale = amplitude_scale**2  # convert nT^2/Hz -> pT^2/Hz
    if gradiometer_amplitude_scale is None:
        gradiometer_amplitude_scale = amplitude_scale
    if gradiometer_psd_scale is None:
        gradiometer_psd_scale = psd_scale

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
            gradiometer_psd_ave_plot = gradiometer_psd_ave[positive_freq_mask] * gradiometer_psd_scale
            if gradiometer_plot_asd:
                gradiometer_psd_ave_plot = np.sqrt(np.maximum(gradiometer_psd_ave_plot, 0.0))
            ax[next_plot].loglog(f_plot, gradiometer_psd_ave_plot, linewidth=2.2)
        gradiometer_psd_plot = gradiometer_psd_theory[positive_freq_mask] * gradiometer_psd_scale
        if gradiometer_plot_asd:
            gradiometer_psd_plot = np.sqrt(np.maximum(gradiometer_psd_plot, 0.0))
        ax[next_plot].loglog(f_plot, gradiometer_psd_plot, c="red")
        if gradiometer_rms_band is not None:
            f_max, rms_label = gradiometer_rms_band
            f_min = float(np.min(f_plot))
            ax[next_plot].axvspan(f_min, f_max, color="tab:blue", alpha=0.15, zorder=0)
            rms_label = str(rms_label).replace("\\n", " ")
            ax[next_plot].text(
                f_min,
                np.max(gradiometer_psd_plot) * 0.8,
                rms_label,
                color="black",
                fontsize=9,
                ha="left",
                va="center",
            )
        if gradiometer_points is not None:
            grad_title = f"Gradiometer Noise ({gradiometer_points} Sensors)"
        else:
            grad_title = "Gradiometer Noise"
        ax[next_plot].set_title(grad_title)
        ax[next_plot].set_xlabel("Frequency [Hz]")
        ax[next_plot].set_ylabel(gradiometer_psd_label)
        if gradiometer_plot_asd:
            ymin = 1e-5
            data_min = float(np.min(gradiometer_psd_plot)) if gradiometer_psd_plot.size else ymin
            if data_min < ymin:
                ax[next_plot].set_ylim(bottom=ymin)
        next_plot += 1

    # Time domain
    ax[next_plot].plot(t, noise_t * amplitude_scale)
    ax[next_plot].set_title("Solar Wind Noise")
    ax[next_plot].set_xlabel("Time [s]")
    ax[next_plot].set_ylabel("Amplitude [pT]")
    if position is not None and len(position) == len(t):
        dist_axis = ax[next_plot].twiny()
        dist_axis.set_xlim(float(position[0]) / 1e3, float(position[-1]) / 1e3)
        dist_axis.set_xlabel("Distance [km]")
    next_plot += 1

    # Sensor-referred ASD (pT/√Hz), assuming uncorrelated sensor noise
    if gradiometer_psd_theory is not None and gradiometer_points is not None:
        if gradiometer_psd_ave is not None:
            psd_use = gradiometer_psd_ave[positive_freq_mask] * gradiometer_psd_scale
        else:
            psd_use = gradiometer_psd_theory[positive_freq_mask] * gradiometer_psd_scale

        asd_out = np.sqrt(np.maximum(psd_use, 0.0))
        length = float(gradiometer_length) if gradiometer_length is not None else 1.0
        if gradiometer_points == 1:
            weight_sq_sum = 1.0
        elif gradiometer_points == 2:
            weight_sq_sum = 2.0
        else:
            _, coeffs = _central_difference_coefficients(gradiometer_points)
            weight_sq_sum = float(np.sum(coeffs**2))
        if gradiometer_output_quantity == "gradient" and length > 0:
            weight_sq_sum = weight_sq_sum / (length**2)
        sensor_asd = asd_out / np.sqrt(weight_sq_sum) if weight_sq_sum > 0 else asd_out
        sensor_asd = sensor_asd * 1e3  # pT/√Hz -> fT/√Hz

        ax[next_plot].loglog(f_plot, sensor_asd, linewidth=2.2)
        sensor_band = sensor_rms_band if sensor_rms_band is not None else gradiometer_rms_band
        if sensor_band is not None:
            f_max, rms_label = sensor_band
            f_min = float(np.min(f_plot))
            ax[next_plot].axvspan(f_min, f_max, color="tab:blue", alpha=0.15, zorder=0)
            rms_label = str(rms_label).replace("\\n", " ")
            ax[next_plot].text(
                f_min,
                np.max(sensor_asd) * 0.8,
                rms_label,
                color="black",
                fontsize=9,
                ha="left",
                va="center",
            )
        ax[next_plot].set_title("Sensor-referred ASD")
        ax[next_plot].set_xlabel("Frequency [Hz]")
        ax[next_plot].set_ylabel("ASD [fT/√Hz]")
        ymin = 0.1
        data_min = float(np.min(sensor_asd)) if sensor_asd.size else ymin
        if data_min < ymin:
            ax[next_plot].set_ylim(bottom=ymin)
        next_plot += 1

    if gradiometer_noise_t is not None and gradiometer_t is not None:
        ax[next_plot].plot(gradiometer_t, gradiometer_noise_t * gradiometer_amplitude_scale)
        ax[next_plot].set_title("Gradiometer Noise Time Domain (Last Realization)")
        ax[next_plot].set_xlabel("Time [s]")
        ax[next_plot].set_ylabel(gradiometer_amplitude_label)
        next_plot += 1

    for i in range(next_plot, len(ax)):
        ax[i].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_power_law_demo(
    frequencies,
    delta_B2_1,
    delta_B2_2,
    delta_B2_3,
    combined_curve,
    point_1,
    point_2,
    point_3,
    alpha1,
    alpha2,
    alpha3,
):
    plt.figure(figsize=(10, 6))
    plt.loglog(frequencies, delta_B2_1, label=f"α={alpha1}", color="red")
    plt.loglog(frequencies, delta_B2_2, label=f"α={alpha2}", color="green")
    plt.loglog(frequencies, delta_B2_3, label=f"α={alpha3}", color="blue")
    plt.loglog(frequencies, combined_curve, label="Combined", color="black", linestyle="--")
    plt.scatter(*point_1, color="red", s=100, marker="o")
    plt.scatter(*point_2, color="green", s=100, marker="o")
    plt.scatter(*point_3, color="blue", s=100, marker="o")
    plt.title("Trace spectrum of magnetic field")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("δB^2 [nT² Hz⁻¹]")
    plt.legend()
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.show()


