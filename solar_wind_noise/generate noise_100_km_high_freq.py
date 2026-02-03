"""Generate a solar-wind noise realization over a gradiometer transit window.

Note: This script does not apply the gradiometer subtraction/filtering; it only
generates the underlying noise field and its PSD.
"""

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


europa = EuropaProperties()
magnetosonic_velocity = europa.magnetosonic_velocity
Europa_transit_time = europa.transit_time
 

gradiometer_length=100e3  # m
gradiometer_points = 2  # odd number of points for gradiometer PSD model
gradiometer_output_quantity = "gradient"  # report gradient in nT/m (later plotted as pT/m)

sample_period=gradiometer_length/magnetosonic_velocity
unity_frequency = magnetosonic_velocity / (6.0 * gradiometer_length)
noise_period = 10.0 / unity_frequency  # so df = 1/T = unity_frequency / 10
sample_frequency=1/sample_period*2**10

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

plot_mask = t <= sample_period
t_plot = t[plot_mask]
noise_plot = noise_t[plot_mask]
position_plot = t_plot * magnetosonic_velocity
grad_plot_mask = t_grad <= sample_period
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
