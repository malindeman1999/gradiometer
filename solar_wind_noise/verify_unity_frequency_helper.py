"""Helper app to verify where 2-point gradiometer PSD matches solar-wind PSD.

For number_of_points == 2, the transfer power is:
    |H(f)|^2 = 4 sin^2(pi f tau), tau = L / v
Unity transfer occurs when |H(f)|^2 = 1.
"""

import numpy as np

from solar_wind_functions import (
    EuropaProperties,
    gradiometer_solar_wind_function,
    solar_wind_function,
)


NUMBER_OF_POINTS = 2
TARGET_BIN_INDEX = 128  # Higher -> narrower bin width around unity frequency.
LENGTH_CONFIGS = [
    {"label": "10 m", "length_m": 10.0},
    {"label": "100 km", "length_m": 100e3},
]


def first_unity_frequency(length_m, velocity_m_per_s):
    """First positive frequency where a 2-point difference has |H|^2 = 1."""
    tau = length_m / velocity_m_per_s
    return 1.0 / (6.0 * tau)


def choose_record_length_for_target_bin(target_f, sample_period, target_bin_index):
    """Choose T so target_f lands on a bin with narrow bandwidth (small df)."""
    sample_frequency = (1.0 / sample_period) * 2**10
    samples_per_target_bin = sample_frequency / target_f
    samples_per_target_bin_int = int(round(samples_per_target_bin))
    if not np.isclose(samples_per_target_bin, samples_per_target_bin_int, rtol=0, atol=1e-9):
        raise ValueError(
            "Could not map target frequency cleanly to a discrete bin "
            "with the current sample frequency."
        )
    n_samples = target_bin_index * samples_per_target_bin_int
    noise_period = n_samples / sample_frequency
    freqs = np.fft.fftfreq(n_samples, 1.0 / sample_frequency)
    positive_freqs = freqs[freqs > 0]
    f_bin = positive_freqs[target_bin_index - 1]
    return f_bin, sample_frequency, noise_period, n_samples


def report_for_length(length_label, length_m, velocity_m_per_s):
    sample_period = length_m / velocity_m_per_s
    f_expected = first_unity_frequency(length_m, velocity_m_per_s)

    f_eval = np.array([f_expected], dtype=float)
    psd_sw_exact = solar_wind_function(f_eval, df=1.0)[0]
    psd_grad_exact = gradiometer_solar_wind_function(
        f=f_eval,
        df=1.0,
        gradiometer_length=length_m,
        magnetosonic_velocity=velocity_m_per_s,
        number_of_points=NUMBER_OF_POINTS,
    )[0]
    ratio_exact = psd_grad_exact / psd_sw_exact

    f_bin, fs, t_record, n_samples = choose_record_length_for_target_bin(
        target_f=f_expected,
        sample_period=sample_period,
        target_bin_index=TARGET_BIN_INDEX,
    )
    f_bin_eval = np.array([f_bin], dtype=float)
    psd_sw_bin = solar_wind_function(f_bin_eval, df=1.0 / t_record)[0]
    psd_grad_bin = gradiometer_solar_wind_function(
        f=f_bin_eval,
        df=1.0 / t_record,
        gradiometer_length=length_m,
        magnetosonic_velocity=velocity_m_per_s,
        number_of_points=NUMBER_OF_POINTS,
    )[0]
    ratio_bin = psd_grad_bin / psd_sw_bin
    rel_diff_bin = abs(psd_grad_bin - psd_sw_bin) / psd_sw_bin
    df = 1.0 / t_record

    print(f"\n=== {length_label} (L = {length_m:g} m) ===")
    print(f"Expected unity frequency (first): {f_expected:.9g} Hz")
    print(f"Exact evaluation ratio PSD_grad / PSD_sw: {ratio_exact:.9g}")
    print(
        f"Chosen FFT bin: {f_bin:.9g} Hz "
        f"(fs={fs:.9g} Hz, T={t_record:.9g} s, N={n_samples}, bin_index={TARGET_BIN_INDEX})"
    )
    print(f"Bin bandwidth: df={df:.9g} Hz (half-band ~ +/- {0.5 * df:.9g} Hz)")
    print(f"Bin ratio PSD_grad / PSD_sw: {ratio_bin:.9g}")
    print(f"Bin relative difference: {100.0 * rel_diff_bin:.4f}%")


def main():
    europa = EuropaProperties()
    v = europa.magnetosonic_velocity
    print("Verifying PSD closeness at expected unity-transfer frequency")
    print(f"Assumed gradiometer model points: {NUMBER_OF_POINTS}")
    for config in LENGTH_CONFIGS:
        report_for_length(
            length_label=config["label"],
            length_m=config["length_m"],
            velocity_m_per_s=v,
        )


if __name__ == "__main__":
    main()
