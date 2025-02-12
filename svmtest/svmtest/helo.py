import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

# Constants
c = 343  # Speed of sound (m/s)
fs = 16000  # Sampling frequency (Hz)
d = 0.021  # Distance between microphones in meters
freq = 1000  # Frequency of interest (Hz)
mic_positions = np.array([[0, 0], [0, d], [d, 0], [d, d]])  # 4-mic square

# Steering Vector Calculation
def steering_vector(angle, freq, mic_positions):
    wavelength = c / freq
    angle_rad = np.deg2rad(angle)
    direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    delays = np.dot(mic_positions, direction) / c
    return np.exp(-1j * 2 * np.pi * freq * delays)

# MUSIC Algorithm
def music(doa_signals, mic_positions, freq, num_sources=1):
    R = np.dot(doa_signals, doa_signals.conj().T) / doa_signals.shape[1]  # Covariance matrix
    eigenvalues, eigenvectors = eig(R)  # Eigen decomposition
    noise_subspace = eigenvectors[:, np.argsort(eigenvalues)[:-num_sources]]  # Noise subspace

    angles = np.arange(0, 181, 1)  # Search angles (0 to 180 degrees)
    pseudospectrum = []
    for angle in angles:
        sv = steering_vector(angle, freq, mic_positions)
        pseudospectrum.append(1 / np.abs(np.dot(sv.conj().T, np.dot(noise_subspace, noise_subspace.conj().T).dot(sv))))

    pseudospectrum = np.array(pseudospectrum)
    return angles, pseudospectrum

# Simulating Signals for Testing
def simulate_signals(num_samples, mic_positions, source_angle, freq, fs):
    t = np.linspace(0, num_samples / fs, num_samples, endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)  # Simulated tone
    wavelength = c / freq
    direction = np.array([np.cos(np.deg2rad(source_angle)), np.sin(np.deg2rad(source_angle))])
    delays = np.dot(mic_positions, direction) / c
    delayed_signals = np.array([np.roll(signal, int(delay * fs)) for delay in delays])
    noise = np.random.normal(0, 0.01, delayed_signals.shape)  # Add some noise
    return delayed_signals + noise

# Main Execution
if __name__ == "__main__":
    # Number of samples for simulation
    num_samples = 16000  # 1 second of data at 16 kHz
    source_angle = 45  # Source angle in degrees

    # Simulate signals arriving at microphones
    data = simulate_signals(num_samples, mic_positions, source_angle, freq, fs)

    # Transpose data for MUSIC (row = microphones, column = time samples)
    doa_signals = data

    # Calculate MUSIC pseudospectrum
    angles, pseudospectrum = music(doa_signals, mic_positions, freq)

    # Plot the pseudospectrum
    plt.figure(figsize=(8, 6))
    plt.plot(angles, 10 * np.log10(pseudospectrum))
    plt.title("MUSIC Pseudospectrum")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Power (dB)")
    plt.grid()
    plt.show()

    print(f"Direction of Arrival (Estimated): {angles[np.argmax(pseudospectrum)]} degrees")
