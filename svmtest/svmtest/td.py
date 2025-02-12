import numpy as np
from scipy.fftpack import fft, ifft, fftshift

# Constants
WINDOW_SIZE = 1024
INTERPOLATION = 10
MAX_TAU = 10  # Adjust based on microphone separation (in samples)

def gcc_phat(sig1, sig2, fs, max_tau=10, interp=10):
    """
    Compute the TDOA between two signals using GCC-PHAT.
    
    Args:
        sig1: Signal from mic 1.
        sig2: Signal from mic 2.
        fs: Sampling frequency (Hz).
        max_tau: Maximum allowable time delay (seconds).
        interp: Interpolation factor for finer TDOA estimation.
    
    Returns:
        TDOA (seconds): Estimated time difference of arrival.
    """
    n = len(sig1) + len(sig2)
    nfft = 2 ** int(np.ceil(np.log2(n))) * interp

    # FFT of signals
    fft_sig1 = fft(sig1, n=nfft)
    fft_sig2 = fft(sig2, n=nfft)

    # Cross-power spectrum
    cross_spectrum = fft_sig1 * np.conj(fft_sig2)
    cross_spectrum /= np.abs(cross_spectrum)

    # Inverse FFT to obtain cross-correlation
    cross_corr = np.real(ifft(cross_spectrum, n=nfft))

    # Shift and wrap the correlation
    cross_corr = fftshift(cross_corr)
    max_shift = int(nfft / 2)
    cross_corr = cross_corr[max_shift - max_tau:max_shift + max_tau]

    # Find the maximum correlation index
    max_idx = np.argmax(cross_corr)
    max_shifted_idx = max_idx - max_tau

    # Compute TDOA
    tdoa = max_shifted_idx / (fs * interp)
    return tdoa

def process_audio(data1, data2, fs):
    """
    Process audio data from two microphones.
    
    Args:
        data1: Audio samples from mic 1.
        data2: Audio samples from mic 2.
        fs: Sampling frequency (Hz).
    
    Returns:
        tdoa: Time difference of arrival (seconds).
    """
    # Convert raw byte data to int16
    mic1_samples = np.frombuffer(data1, dtype=np.int16)
    mic2_samples = np.frombuffer(data2, dtype=np.int16)

    # Ensure same length for both signals
    min_length = min(len(mic1_samples), len(mic2_samples))
    mic1_samples = mic1_samples[:min_length]
    mic2_samples = mic2_samples[:min_length]

    # Compute TDOA using GCC-PHAT
    tdoa = gcc_phat(mic1_samples, mic2_samples, fs, max_tau=MAX_TAU * INTERPOLATION, interp=INTERPOLATION)
    return tdoa

# Example usage for real-time processing
import serial

# Serial port configurations (adjust COM ports as needed)
master_port = serial.Serial("COM10", 1000000, timeout=1)
slave_port = serial.Serial("COM13", 1000000, timeout=1)

fs = 48000  # Sampling frequency in Hz

try:
    print("Processing real-time audio data...")
    while True:
        # Read data from both serial ports
        data1 = master_port.read(WINDOW_SIZE * 2)  # 16-bit (2 bytes) per sample
        data2 = slave_port.read(WINDOW_SIZE * 2)

        if data1 and data2:
            tdoa = process_audio(data1, data2, fs)
            print(f"TDOA: {tdoa:.12f} seconds")
except KeyboardInterrupt:
    master_port.close()
    slave_port.close()
    print("Processing stopped.")
