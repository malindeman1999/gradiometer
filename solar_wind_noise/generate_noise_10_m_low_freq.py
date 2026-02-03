"""Generate a low-frequency solar-wind noise realization for a 10 m gradiometer.

Note: This script does not apply the gradiometer subtraction/filtering; it only
generates the underlying noise field and its PSD.
"""

import matplotlib.pyplot as plt
import numpy as np

from solar_wind_functions import (
    compute_average_psd,
    EuropaProperties,
    gradiometer_solar_wind_function,
    gradiometer_transfer_power,
    plot_noise_results,
    solar_wind_function,
    summarize_noise_results,
    uniform_PSD,
)









# func = uniform_PSD
func = solar_wind_function


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


europa = EuropaProperties()
magnetosonic_velocity = europa.magnetosonic_velocity
Europa_transit_time = europa.transit_time
 

gradiometer_length=10  # m
gradiometer_points = 5  # odd number of points for gradiometer PSD model
gradiometer_output_quantity = "gradient"  # report gradient in nT/m (later plotted as pT/m)

sample_period = 0.01  # 10 ms sampling cadence
sample_frequency = 1.0 / sample_period
noise_period = 3600.0  # 1 hour so lowest frequency is 1 / 1 hour

print(f"Europa transit time: {Europa_transit_time} s")


number_of_samples=30
realization_seed = 12345
noise_t, t, noise_f, f, normalization, PSD, df, PSD_ave, diff_rms = compute_average_psd(
    T_sample=noise_period,
    fs_sample=sample_frequency,
    number_of_samples=number_of_samples,
    func=func,
    seed=realization_seed,
)
(
    noise_t_grad,
    t_grad,
    _noise_f_grad,
    _f_grad,
    _normalization_grad,
    _PSD_grad,
    _df_grad,
    gradiometer_psd_ave,
    _diff_rms_grad,
) = compute_average_psd(
    T_sample=noise_period,
    fs_sample=sample_frequency,
    number_of_samples=number_of_samples,
    seed=realization_seed,
    func=lambda freq, delta_f: gradiometer_solar_wind_function(
        f=freq,
        df=delta_f,
        gradiometer_length=gradiometer_length,
        magnetosonic_velocity=magnetosonic_velocity,
        number_of_points=gradiometer_points,
        output_quantity=gradiometer_output_quantity,
    ),
)

PSD_theory, position = summarize_noise_results(
    f=f,
    df=df,
    PSD=PSD,
    PSD_ave=PSD_ave,
    noise_t=noise_t,
    t=t,
    magnetosonic_velocity=magnetosonic_velocity,
    func=func,
    diff_rms=diff_rms,
)

plot_mask = t <= noise_period
t_plot = t[plot_mask]
noise_plot = noise_t[plot_mask]
position_plot = t_plot * magnetosonic_velocity
grad_plot_mask = t_grad <= noise_period
t_grad_plot = t_grad[grad_plot_mask]
noise_grad_plot = noise_t_grad[grad_plot_mask]
gradiometer_psd_theory = gradiometer_solar_wind_function(
    f=f,
    df=df,
    gradiometer_length=gradiometer_length,
    magnetosonic_velocity=magnetosonic_velocity,
    number_of_points=gradiometer_points,
    output_quantity=gradiometer_output_quantity,
)
gradiometer_transfer_theory = gradiometer_transfer_power(
    f=f,
    gradiometer_length=gradiometer_length,
    magnetosonic_velocity=magnetosonic_velocity,
    number_of_points=gradiometer_points,
    output_quantity=gradiometer_output_quantity,
)

gradiometer_amplitude_scale = 1e3  # nT/m -> pT/m
gradiometer_psd_scale = gradiometer_amplitude_scale**2  # (nT/m)^2/Hz -> (pT/m)^2/Hz

plot_noise_results(
    f=f,
    PSD_ave=PSD_ave,
    PSD_theory=PSD_theory,
    t=t_plot,
    noise_t=noise_plot,
    position=position_plot,
    gradiometer_psd_ave=gradiometer_psd_ave,
    gradiometer_psd_theory=gradiometer_psd_theory,
    gradiometer_transfer_power=gradiometer_transfer_theory,
    gradiometer_transfer_frequency=f,
    gradiometer_noise_t=noise_grad_plot,
    gradiometer_t=t_grad_plot,
    gradiometer_points=gradiometer_points,
    gradiometer_amplitude_scale=gradiometer_amplitude_scale,
    gradiometer_psd_scale=gradiometer_psd_scale,
    gradiometer_amplitude_label="Amplitude [pT/m]",
    gradiometer_psd_label="ASD [pT/(m*Hz^0.5)]",
    gradiometer_plot_asd=True,
)

lag_noise, corr_noise, err_noise = centered_correlation(noise_t, sample_period)
lag_grad, corr_grad, err_grad = centered_correlation(noise_t_grad, sample_period)
max_lag_seconds = 300.0
positive_lag_mask = (lag_noise > 0) & (lag_noise <= max_lag_seconds)

plt.figure(figsize=(10, 4))
plt.errorbar(
    lag_noise[positive_lag_mask],
    corr_noise[positive_lag_mask],
    yerr=err_noise[positive_lag_mask],
    fmt="o",
    markersize=2.5,
    elinewidth=0.7,
    capsize=0,
    linestyle="none",
    alpha=0.8,
    label="Solar wind noise",
)
plt.errorbar(
    lag_grad[positive_lag_mask],
    corr_grad[positive_lag_mask],
    yerr=err_grad[positive_lag_mask],
    fmt="o",
    markersize=2.5,
    elinewidth=0.7,
    capsize=0,
    linestyle="none",
    alpha=0.8,
    label="Gradiometer noise",
)
plt.xscale("log")
plt.title("Centered Correlation Function")
plt.xlabel("Positive lag [s]")
plt.ylabel("Correlation [1]")
plt.ylim(-1.0, 1.0)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

