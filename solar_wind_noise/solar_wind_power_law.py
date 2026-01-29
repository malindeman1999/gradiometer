import numpy as np
import matplotlib.pyplot as plt

def power_law(f, alpha, C=1):
    """Power law function."""
    return C * f**alpha

def calculate_C(delta_B2, f, alpha):
    """Calculate the normalization constant C based on a known point and alpha."""
    return delta_B2 / (f**alpha)
def combined_power_law(frequencies_signed, alpha1, C1, alpha2, C2, alpha3, C3):
    """Compute the combined power law for a set of frequencies.
    unit nT*2/ Hz
    
    """
    frequencies=np.abs(frequencies_signed)  # fft will return negative frequencies
    

    delta_B2_1 = power_law(frequencies, alpha1, C1)
    delta_B2_2 = power_law(frequencies, alpha2, C2)
    delta_B2_3 = power_law(frequencies, alpha3, C3)
    
    # Return the element-wise smallest values among the three
    return np.minimum.reduce([delta_B2_1, delta_B2_2, delta_B2_3])


if __name__ == "__main__":
    # Given points from the plot
    point_1 = (0.004, 1.5*10**4)  # for alpha = -1.46
    point_2 = (0.004, 1.5*10**4)   # for alpha = -1.64
    point_3 = (0.4, 10.)    # for alpha = -2.57

    # Calculate power spectra for the three alphas
    alpha1, alpha2, alpha3 = -1.46, -1.64, -2.57

    # Calculate C values
    C1 = calculate_C(point_1[1], point_1[0], alpha1)
    C2 = calculate_C(point_2[1], point_2[0], alpha2)
    C3 = calculate_C(point_3[1], point_3[0], alpha3)
    print(f"C values {C1}, {C2}, {C3}")

    # Define frequency range
    frequencies = np.logspace(-5, 1, 1000)  # From the plot: 0.00001 to 10 Hz

    delta_B2_1 = power_law(frequencies, alpha1, C1)
    delta_B2_2 = power_law(frequencies, alpha2, C2)
    delta_B2_3 = power_law(frequencies, alpha3, C3)

    # Combine the three curves by taking the minimum value for each frequency
    combined_curve = combined_power_law(frequencies, alpha1, C1, alpha2, C2, alpha3, C3)
    # Plot
    plt.figure(figsize=(10, 6))
    plt.loglog(frequencies, delta_B2_1, label=f"α={alpha1}", color='red')
    plt.loglog(frequencies, delta_B2_2, label=f"α={alpha2}", color='green')
    plt.loglog(frequencies, delta_B2_3, label=f"α={alpha3}", color='blue')
    plt.loglog(frequencies, combined_curve, label="Combined", color='black', linestyle='--')

    # Plot the estimation points
    plt.scatter(*point_1, color='red', s=100, marker='o')
    plt.scatter(*point_2, color='green', s=100, marker='o')
    plt.scatter(*point_3, color='blue', s=100, marker='o')

    plt.title("Trace spectrum of magnetic field")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("δB^2 [nT² Hz⁻¹]")
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.show()

