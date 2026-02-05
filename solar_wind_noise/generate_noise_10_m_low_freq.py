"""Generate a low-frequency solar-wind noise realization for a 10 m gradiometer.

Note: This script does not apply the gradiometer subtraction/filtering; it only
generates the underlying noise field and its PSD.
"""

import numpy as np
import math

from solar_wind_functions import (
    compute_average_psd,
    central_difference_coefficients,
    EuropaProperties,
    gradiometer_solar_wind_function,
    gradiometer_transfer_power,
    solar_wind_function,
    summarize_noise_results,
    uniform_PSD,
)
from plot_routines import plot_noise_results, centered_correlation, plot_centered_correlation









# func = uniform_PSD
func = solar_wind_function


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

crossing_time_s = 114.483
freq_max = 1.0 / crossing_time_s
print(f"Node transit time: {crossing_time_s:.3f} s")
print(f"RMS frequency range: 0 to {freq_max:.6e} Hz")
band_mask = (f > 0.0) & (f <= freq_max)
if np.any(band_mask):
    band_variance = float(np.sum(gradiometer_psd_ave[band_mask]) * df)
    band_rms = math.sqrt(max(band_variance, 0.0)) * gradiometer_amplitude_scale * 1e3
    band_label = f"RMS = {band_rms:.1f} fT/m"
    print(f"RMS noise: {band_rms:.1f} fT/m")
    if gradiometer_points == 1:
        weight_sq_sum = 1.0
    elif gradiometer_points == 2:
        weight_sq_sum = 2.0
    else:
        _, coeffs = central_difference_coefficients(gradiometer_points)
        weight_sq_sum = float(np.sum(coeffs**2))
    if gradiometer_output_quantity == "gradient":
        weight_sq_sum = weight_sq_sum / (gradiometer_length**2)
    sensor_psd_band = gradiometer_psd_ave[band_mask] * gradiometer_psd_scale
    if weight_sq_sum > 0:
        sensor_psd_band = sensor_psd_band / weight_sq_sum
    sensor_rms = math.sqrt(float(np.sum(sensor_psd_band) * df)) if sensor_psd_band.size else float("nan")
    sensor_rms_ft = sensor_rms * 1e3
    sensor_label = f"RMS = {sensor_rms_ft:.1f} fT"
else:
    band_label = "RMS = N/A"
    sensor_label = "RMS = N/A"
    print("RMS noise: N/A")

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
    gradiometer_length=gradiometer_length,
    gradiometer_output_quantity=gradiometer_output_quantity,
    gradiometer_amplitude_scale=gradiometer_amplitude_scale,
    gradiometer_psd_scale=gradiometer_psd_scale,
    gradiometer_amplitude_label="Amplitude [pT/m]",
    gradiometer_psd_label="ASD [pT/(m*√Hz)]",
    gradiometer_plot_asd=True,
    gradiometer_rms_band=(freq_max, band_label),
    sensor_rms_band=(freq_max, sensor_label),
)

print("Computing centered correlation...")
lag_noise, corr_noise, err_noise = centered_correlation(noise_t, sample_period)
lag_grad, corr_grad, err_grad = centered_correlation(noise_t_grad, sample_period)
plot_centered_correlation(
    lag_noise,
    corr_noise,
    err_noise,
    lag_grad,
    corr_grad,
    err_grad,
    max_lag_seconds=300.0,
)

