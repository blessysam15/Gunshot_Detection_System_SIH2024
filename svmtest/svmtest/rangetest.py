import pyaudio
import numpy as np
import scipy.signal as signal

# Constants for Range Estimation
ALPHA_CONSTANT = 0.005  # Atmospheric attenuation constant (dB/m)
NOISE_LEVEL_DB = 40  # Estimated noise level in dB
DETECTION_THRESHOLD_DB = 10  # Detection threshold in dB
SAMPLE_RATE = 44100  # Sampling rate in Hz
CHUNK_SIZE = 2048  # Size of audio chunks

# Frequency bands to analyze (drone noise frequencies)
FREQ_BANDS = [500, 1000, 1500]  # Hz (e.g., drone sound fundamental frequencies)

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE)

def calculate_rms(data):
    """Calculate RMS (Root Mean Square) of audio signal."""
    return np.sqrt(np.mean(np.square(data)))

def calculate_source_level(rms):
    """Calculate source level (SL) from RMS."""
    # Convert RMS to decibels
    return 20 * np.log10(rms + 1e-8)

def estimate_range(sl, nl, dt, alpha):
    """Estimate range using the passive sonar equation."""
    transmission_loss = sl - nl - dt
    if transmission_loss <= 0:
        return 0  # No detection possible
    return transmission_loss / alpha

def process_audio():
    print("Listening for drone sounds...")
    while True:
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.int16)

        # Compute RMS and source level for each frequency band
        for freq in FREQ_BANDS:
            sos = signal.butter(4, [freq - 50, freq + 50], btype='band', fs=SAMPLE_RATE, output='sos')
            filtered = signal.sosfilt(sos, audio)
            rms = calculate_rms(filtered)
            source_level = calculate_source_level(rms)

            # Estimate detection range
            range_estimation = estimate_range(source_level, NOISE_LEVEL_DB, DETECTION_THRESHOLD_DB, ALPHA_CONSTANT)

            print(f"Frequency: {freq} Hz | Source Level: {source_level:.2f} dB | Estimated Range: {range_estimation:.2f} m")

try:
    process_audio()
except KeyboardInterrupt:
    print("Stopping audio processing...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
