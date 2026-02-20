"""Unified GUI runner for solar-wind noise generation presets.

This combines the four preset scripts into one app:
1) 10 m low frequency (default)
2) 10 m high frequency
3) 100 km low frequency
4) 100 km high frequency
"""

from __future__ import annotations

import math
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk

import numpy as np

from plot_routines import (
    centered_correlation,
    plot_centered_correlation,
    plot_noise_results,
)
from solar_wind_functions import (
    EuropaProperties,
    central_difference_coefficients,
    compute_average_psd,
    gradiometer_solar_wind_function,
    gradiometer_transfer_power,
    solar_wind_function,
    summarize_noise_results,
)

# Default from oribital/orbit_estimates.py estimate_default_orbit().node_transit_time_s
# (used historically as the RMS-band upper-limit timescale via f_max = 1 / transit_time).
DEFAULT_GROUND_TRANSIT_TIME_S = 114.483


@dataclass(frozen=True)
class NoisePreset:
    key: str
    label: str
    gradiometer_length_m: float
    gradiometer_points: int
    sample_period_s: float
    sample_frequency_hz: float
    noise_period_s: float
    plot_time_window_s: float
    show_centered_correlation: bool
    correlation_max_lag_s: float | None
    show_rms_band: bool


def _build_presets(magnetosonic_velocity: float) -> dict[str, NoisePreset]:
    length_10m = 10.0
    sp_10m_high = length_10m / magnetosonic_velocity
    uf_10m = magnetosonic_velocity / (6.0 * length_10m)
    np_10m_high = 10.0 / uf_10m
    sf_10m_high = (1.0 / sp_10m_high) * (2**10)

    length_100km = 100e3
    sp_100km_high = length_100km / magnetosonic_velocity
    uf_100km = magnetosonic_velocity / (6.0 * length_100km)
    np_100km_high = 10.0 / uf_100km
    sf_100km_high = (1.0 / sp_100km_high) * (2**10)

    return {
        "10m_low": NoisePreset(
            key="10m_low",
            label="10 m low frequency",
            gradiometer_length_m=length_10m,
            gradiometer_points=2,
            sample_period_s=0.01,
            sample_frequency_hz=100.0,
            noise_period_s=3600.0,
            plot_time_window_s=3600.0,
            show_centered_correlation=True,
            correlation_max_lag_s=300.0,
            show_rms_band=True,
        ),
        "10m_high": NoisePreset(
            key="10m_high",
            label="10 m high frequency",
            gradiometer_length_m=length_10m,
            gradiometer_points=5,
            sample_period_s=sp_10m_high,
            sample_frequency_hz=sf_10m_high,
            noise_period_s=np_10m_high,
            plot_time_window_s=sp_10m_high,
            show_centered_correlation=False,
            correlation_max_lag_s=None,
            show_rms_band=False,
        ),
        "100km_low": NoisePreset(
            key="100km_low",
            label="100 km low frequency",
            gradiometer_length_m=length_100km,
            gradiometer_points=2,
            sample_period_s=1.0 / 200.0,
            sample_frequency_hz=200.0,
            noise_period_s=1000.0,
            plot_time_window_s=1000.0,
            show_centered_correlation=True,
            correlation_max_lag_s=None,
            show_rms_band=True,
        ),
        "100km_high": NoisePreset(
            key="100km_high",
            label="100 km high frequency",
            gradiometer_length_m=length_100km,
            gradiometer_points=2,
            sample_period_s=sp_100km_high,
            sample_frequency_hz=sf_100km_high,
            noise_period_s=np_100km_high,
            plot_time_window_s=sp_100km_high,
            show_centered_correlation=False,
            correlation_max_lag_s=None,
            show_rms_band=False,
        ),
    }


def _run_preset(
    preset: NoisePreset,
    magnetosonic_velocity: float,
    number_of_samples: int,
    realization_seed: int,
    ground_transit_time_s: float,
    log,
) -> dict | None:
    gradiometer_length = float(preset.gradiometer_length_m)
    gradiometer_points = int(preset.gradiometer_points)
    gradiometer_output_quantity = "gradient"

    log(f"Running preset: {preset.label}")
    log(f"gradiometer_length={gradiometer_length:g} m, points={gradiometer_points}")
    log(
        f"sample_frequency={preset.sample_frequency_hz:.6g} Hz, "
        f"noise_period={preset.noise_period_s:.6g} s"
    )

    noise_t, t, _noise_f, f, _normalization, PSD, df, PSD_ave, diff_rms = compute_average_psd(
        T_sample=preset.noise_period_s,
        fs_sample=preset.sample_frequency_hz,
        number_of_samples=number_of_samples,
        func=solar_wind_function,
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
        T_sample=preset.noise_period_s,
        fs_sample=preset.sample_frequency_hz,
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

    PSD_theory, _position = summarize_noise_results(
        f=f,
        df=df,
        PSD=PSD,
        PSD_ave=PSD_ave,
        noise_t=noise_t,
        t=t,
        magnetosonic_velocity=magnetosonic_velocity,
        func=solar_wind_function,
        diff_rms=diff_rms,
    )

    plot_mask = t <= preset.plot_time_window_s
    t_plot = t[plot_mask]
    noise_plot = noise_t[plot_mask]
    position_plot = t_plot * magnetosonic_velocity
    grad_plot_mask = t_grad <= preset.plot_time_window_s
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

    gradiometer_amplitude_scale = 1e3
    gradiometer_psd_scale = gradiometer_amplitude_scale**2

    band_kwargs = {}
    if preset.show_rms_band:
        crossing_time_s = float(ground_transit_time_s)
        freq_max = 1.0 / crossing_time_s
        log(f"Node transit time: {crossing_time_s:.3f} s")
        log(f"RMS frequency range: 0 to {freq_max:.6e} Hz")
        band_mask = (f > 0.0) & (f <= freq_max)
        if np.any(band_mask):
            band_variance = float(np.sum(gradiometer_psd_ave[band_mask]) * df)
            band_rms = math.sqrt(max(band_variance, 0.0)) * gradiometer_amplitude_scale * 1e3
            band_label = f"RMS = {band_rms:.1f} fT/m"
            log(f"RMS noise: {band_rms:.1f} fT/m")
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
            sensor_rms = (
                math.sqrt(float(np.sum(sensor_psd_band) * df))
                if sensor_psd_band.size
                else float("nan")
            )
            sensor_rms_ft = sensor_rms * 1e3
            sensor_label = f"RMS = {sensor_rms_ft:.1f} fT"
        else:
            band_label = "RMS = N/A"
            sensor_label = "RMS = N/A"
            log("RMS noise: N/A")
        band_kwargs["gradiometer_rms_band"] = (freq_max, band_label)
        band_kwargs["sensor_rms_band"] = (freq_max, sensor_label)

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
        solar_wind_plot_asd=True,
        **band_kwargs,
    )

    correlation_payload = None
    if preset.show_centered_correlation:
        correlation_payload = {
            "noise_t": np.asarray(noise_t, dtype=float),
            "noise_t_grad": np.asarray(noise_t_grad, dtype=float),
            "sample_period_s": float(preset.sample_period_s),
            "max_lag_seconds": preset.correlation_max_lag_s,
        }
        log("Correlation data prepared. Use 'Plot Correlation' to render.")

    log("Done.")
    return correlation_payload


def main() -> None:
    europa = EuropaProperties()
    magnetosonic_velocity = europa.magnetosonic_velocity
    presets = _build_presets(magnetosonic_velocity)

    root = tk.Tk()
    root.title("Solar Wind Noise Presets")
    root.geometry("760x520")

    selected_key = tk.StringVar(value="10m_low")
    samples_var = tk.StringVar(value="30")
    seed_var = tk.StringVar(value="12345")
    ground_transit_var = tk.StringVar(value=f"{DEFAULT_GROUND_TRANSIT_TIME_S:.3f}")
    details_var = tk.StringVar(value="")

    frame = ttk.Frame(root, padding=10)
    frame.pack(fill="both", expand=True)

    ttk.Label(frame, text="Preset").pack(anchor="w")
    for key in ("10m_low", "10m_high", "100km_low", "100km_high"):
        ttk.Radiobutton(
            frame,
            text=presets[key].label,
            value=key,
            variable=selected_key,
        ).pack(anchor="w")

    controls = ttk.Frame(frame)
    controls.pack(fill="x", pady=(12, 8))
    ttk.Label(controls, text="Realizations:").grid(row=0, column=0, sticky="w")
    ttk.Entry(controls, textvariable=samples_var, width=10).grid(row=0, column=1, padx=(6, 16), sticky="w")
    ttk.Label(controls, text="Seed:").grid(row=0, column=2, sticky="w")
    ttk.Entry(controls, textvariable=seed_var, width=14).grid(row=0, column=3, padx=(6, 16), sticky="w")
    ttk.Label(controls, text="Ground transit [s]:").grid(row=1, column=0, sticky="w", pady=(6, 0))
    ttk.Entry(controls, textvariable=ground_transit_var, width=10).grid(row=1, column=1, padx=(6, 16), sticky="w", pady=(6, 0))

    ttk.Label(frame, textvariable=details_var, foreground="#2d2d2d", justify="left").pack(anchor="w", pady=(2, 8))

    log_box = tk.Text(frame, height=14, wrap="word")
    log_box.pack(fill="both", expand=True)
    correlation_payload: dict | None = None

    def log(msg: str) -> None:
        log_box.insert(tk.END, msg + "\n")
        log_box.see(tk.END)
        root.update_idletasks()
        print(msg)

    def refresh_details(*_args) -> None:
        p = presets[selected_key.get()]
        details_var.set(
            f"length={p.gradiometer_length_m:g} m, points={p.gradiometer_points}, "
            f"fs={p.sample_frequency_hz:.6g} Hz, T={p.noise_period_s:.6g} s"
        )

    refresh_details()
    selected_key.trace_add("write", refresh_details)

    def run_selected() -> None:
        nonlocal correlation_payload
        try:
            n_samples = int(samples_var.get().strip())
            seed = int(seed_var.get().strip())
            ground_transit_time_s = float(ground_transit_var.get().strip())
            if n_samples <= 0:
                raise ValueError("Realizations must be > 0.")
            if ground_transit_time_s <= 0.0:
                raise ValueError("Ground transit time must be > 0.")
            preset = presets[selected_key.get()]
            run_btn.state(["disabled"])
            corr_btn.state(["disabled"])
            correlation_payload = _run_preset(
                preset=preset,
                magnetosonic_velocity=magnetosonic_velocity,
                number_of_samples=n_samples,
                realization_seed=seed,
                ground_transit_time_s=ground_transit_time_s,
                log=log,
            )
            if correlation_payload is not None:
                corr_btn.state(["!disabled"])
        except Exception as exc:  # noqa: BLE001
            log(f"Error: {exc}")
            correlation_payload = None
            corr_btn.state(["disabled"])
        finally:
            run_btn.state(["!disabled"])

    def plot_correlation() -> None:
        if correlation_payload is None:
            log("No correlation data available. Run a low-frequency preset first.")
            return
        log("Computing centered correlation...")
        lag_noise, corr_noise, err_noise = centered_correlation(
            correlation_payload["noise_t"],
            correlation_payload["sample_period_s"],
        )
        lag_grad, corr_grad, err_grad = centered_correlation(
            correlation_payload["noise_t_grad"],
            correlation_payload["sample_period_s"],
        )
        plot_centered_correlation(
            lag_noise,
            corr_noise,
            err_noise,
            lag_grad,
            corr_grad,
            err_grad,
            max_lag_seconds=correlation_payload["max_lag_seconds"],
        )

    run_btn = ttk.Button(frame, text="Calculate", command=run_selected)
    run_btn.pack(anchor="w", pady=(8, 0))
    corr_btn = ttk.Button(frame, text="Plot Correlation", command=plot_correlation)
    corr_btn.state(["disabled"])
    corr_btn.pack(anchor="w", pady=(6, 0))

    root.mainloop()


if __name__ == "__main__":
    main()
