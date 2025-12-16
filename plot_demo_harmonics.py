"""
Utility plotting for the demo: stacked bar charts per (l,m) for
|B_r|, |dB/dt|, |E_tor|, |K_tor|, and |B_emit,r|.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

from phasor_data import PhasorSimulation


def _flatten(coeffs: torch.Tensor):
    """Return l, m, |coeff| arrays in canonical (l,m) order."""
    lmax = coeffs.shape[-2] - 1
    ls, ms, mags = [], [], []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            ls.append(l)
            ms.append(m)
            mags.append(torch.abs(coeffs[l, lmax + m]).item())
    return np.array(ls), np.array(ms), np.array(mags)


def plot_demo_harmonics(phasor_sim: PhasorSimulation, save_path: str, eps: float = 1e-15) -> None:
    """
    Stacked bar plot over (l,m):
      row1 |B_r|, row2 |dB/dt| (omega * |B_r|), row3 |E_tor|, row4 |K_tor|, row5 |B_emit,r|.
    X-axis is canonical (l,m), truncated at highest non-zero l (min l=1).
    """
    if phasor_sim.B_radial is None or phasor_sim.E_toroidal is None or phasor_sim.K_toroidal is None:
        raise ValueError("PhasorSimulation missing required phasors for plotting.")

    B_rad_ph = phasor_sim.B_radial
    B_rad_emit_ph = phasor_sim.B_rad_emit if phasor_sim.B_rad_emit is not None else torch.zeros_like(B_rad_ph)
    E_tor_ph = phasor_sim.E_toroidal
    K_tor_ph = phasor_sim.K_toroidal
    omega = float(phasor_sim.omega)

    l_b, m_b, mag_b = _flatten(B_rad_ph)
    _, _, mag_dbdt = l_b, m_b, omega * mag_b
    _, _, mag_e = _flatten(E_tor_ph)
    _, _, mag_k = _flatten(K_tor_ph)
    _, _, mag_bemit = _flatten(B_rad_emit_ph)

    mags_all = np.stack([mag_b, mag_dbdt, mag_e, mag_k, mag_bemit], axis=0)
    nonzero_mask = mags_all > eps
    active_ls = l_b[np.any(nonzero_mask, axis=0)]
    l_cut = int(active_ls.max()) if active_ls.size else 1
    l_cut = max(l_cut, 1)
    keep = l_b <= l_cut

    labels = [f"({l},{m})" for l, m in zip(l_b[keep], m_b[keep])]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(5, 1, figsize=(max(10, len(labels) * 0.3), 12), sharex=True)
    axes[0].bar(x, mag_b[keep], color="#4472c4")
    axes[0].set_ylabel("|B_r|")
    axes[0].set_title("Ambient normal field phasors")

    axes[1].bar(x, mag_dbdt[keep], color="#5c9bd5")
    axes[1].set_ylabel("|dB/dt|")
    axes[1].set_title("Normal dB/dt phasors")

    axes[2].bar(x, mag_e[keep], color="#2ca7a0")
    axes[2].set_ylabel("|E_tor|")
    axes[2].set_title("Toroidal E phasors")

    axes[3].bar(x, mag_k[keep], color="#70ad47")
    axes[3].set_ylabel("|K_tor|")
    axes[3].set_title("Toroidal current phasors")

    axes[4].bar(x, mag_bemit[keep], color="#c55a11")
    axes[4].set_ylabel("|B_emit,r|")
    axes[4].set_title("Emitted normal field phasors")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=90)
    axes[-1].set_xlabel("(l,m) up to active l (min l=1)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Harmonic amplitudes plot saved to {save_path}")
