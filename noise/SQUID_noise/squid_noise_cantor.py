import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def magnetic_field_noise(f_hz, S_white_fT=20.0, A_flicker_at_1Hz_fT=24.0):
    """
    ASD model (fT/√Hz): white + flicker (PSD ∝ 1/f, so ASD ∝ 1/sqrt(f)),
    summed in quadrature.

    Based on:
    Cantor et al., IEEE Trans. Appl. Supercond., Vol. 5, No. 2, June 1995.
    
    The paper says white noise is 10 fT/√Hz, but their Fig. 4 shows ~20 fT/√Hz at medium frequencies. To match the figure, we use 20 fT/√Hz for the white noise level.
    """
    f = np.asarray(f_hz, dtype=float)
    if np.any(f <= 0):
        raise ValueError("All frequencies must be > 0 Hz.")

    S_white = np.full_like(f, float(S_white_fT))
    S_flicker = float(A_flicker_at_1Hz_fT) / np.sqrt(f)
    S_total = np.sqrt(S_white**2 + S_flicker**2)

    return S_white, S_flicker, S_total


# Frequency range: 1 mHz to 10 kHz
f = np.logspace(-3, 4, 2000)

# Match paper endpoints (~10 fT/√Hz white, ~26 fT/√Hz at 1 Hz)
S_white = 20.0  #The paper says white noise is 10 fT/√Hz, but their Fig. 4 shows ~20 fT/√Hz at medium frequencies. To match the figure, we use 20 fT/√Hz for the white noise level.
S_1Hz_total = 26.0
A_1Hz = np.sqrt(S_1Hz_total**2 - S_white**2)
C_1overf_psd = A_1Hz**2  # PSD model: S_B(f) = C_1overf_psd / f (fT^2/Hz)

print(f"1/f ASD coefficient at 1 Hz: A_1Hz = {A_1Hz:.6g} fT/sqrt(Hz)")
print(f"1/f PSD coefficient: C_1overf = {C_1overf_psd:.6g} fT^2")

Sw, Sf, Stot = magnetic_field_noise(f, S_white_fT=S_white, A_flicker_at_1Hz_fT=A_1Hz)

# Load previously digitized Fig. 4 data (1–1000 Hz region)
data_path = Path(__file__).resolve().parent / "cantor1995_fig4_digitized_raw.csv"
df_fig4 = pd.read_csv(data_path)

plt.figure()
plt.loglog(f, Sw, label="White ASD")
plt.loglog(f, Sf, label="Flicker ASD (PSD ∝ 1/f)")
plt.loglog(f, Stot, label="Total ASD (model)")

plt.loglog(df_fig4["f_hz"], df_fig4["S_fT"], ".", label="Cantor et al. Fig. 4 (digitized)")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnetic field noise ASD (fT/√Hz)")
plt.title("Model + Cantor et al. (1995) Fig. 4 Overplot")
plt.grid(True, which="both")
plt.legend()
plt.show()

