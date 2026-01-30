"""Generate a solar-wind noise realization over a gradiometer transit window.

Note: This script does not apply the gradiometer subtraction/filtering; it only
generates the underlying noise field and its PSD.
"""

from solar_wind_functions import (
    apply_gradiometer_difference,
    compute_average_psd,
    EuropaProperties,
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
 

gradiometer_length=10  # m

sample_period=gradiometer_length/magnetosonic_velocity
noise_period=2 * sample_period
sample_frequency=1/sample_period*2**10

print(f"Europa transit time: {Europa_transit_time} s")


number_of_samples=30
noise_t, t, noise_f, f, normalization, PSD, df, PSD_ave, diff_rms = compute_average_psd(
    T_sample=noise_period, fs_sample=sample_frequency, number_of_samples=number_of_samples, func=func
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

diff_t, t_diff, delay_samples = apply_gradiometer_difference(
    noise_t=noise_t, fs=sample_frequency, tau=sample_period
)

plot_mask = t <= sample_period
t_plot = t[plot_mask]
noise_plot = noise_t[plot_mask]
position_plot = t_plot * magnetosonic_velocity
diff_mask = t_diff <= sample_period
diff_plot = diff_t[diff_mask]
t_diff_plot = t_diff[diff_mask]

plot_noise_results(
    f=f,
    PSD_ave=PSD_ave,
    PSD_theory=PSD_theory,
    t=t_plot,
    noise_t=noise_plot,
    position=position_plot,
    diff_t=diff_plot,
    t_diff=t_diff_plot,
)

