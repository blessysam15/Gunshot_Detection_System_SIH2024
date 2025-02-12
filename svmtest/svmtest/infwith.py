import numpy as np
import sounddevice as sd
import scipy.fftpack
import joblib

# Function to extract MFCC-like features without librosa
def extract_features(audio, sr, n_mfcc=13):
    # Normalize audio
    if audio.dtype != np.float32:
        audio = audio / np.max(np.abs(audio), axis=0)

    # Perform a Short-Time Fourier Transform (STFT)
    window_size = 1024
    hop_size = 512
    window = np.hamming(window_size)
    num_frames = 1 + (len(audio) - window_size) // hop_size
    frames = np.lib.stride_tricks.sliding_window_view(audio, window_size)[::hop_size].copy()  # Make writable
    frames = frames.astype(np.float32)  # Ensure writable dtype
    frames *= window
    stft = np.abs(np.fft.rfft(frames, axis=1))

    # Compute mel filterbanks
    mel_filters = np.zeros((n_mfcc, stft.shape[1]))
    mel_freq = np.linspace(0, sr / 2, stft.shape[1])
    mel_points = np.linspace(0, sr / 2, n_mfcc + 2)
    for i in range(1, len(mel_points) - 1):
        mel_filters[i - 1, :] = np.maximum(0, np.minimum(
            (mel_freq - mel_points[i - 1]) / (mel_points[i] - mel_points[i - 1]),
            (mel_points[i + 1] - mel_freq) / (mel_points[i + 1] - mel_points[i])
        ))

    # Apply mel filters
    mel_spectrum = np.dot(stft, mel_filters.T)

    # Compute logarithm and discrete cosine transform (DCT)
    log_mel_spectrum = np.log(np.maximum(mel_spectrum, 1e-10))
    mfcc = scipy.fftpack.dct(log_mel_spectrum, axis=1)[:, :n_mfcc]

    # Return mean of MFCCs across frames
    return np.mean(mfcc, axis=0)

# Function to capture audio from the microphone
def capture_audio(duration, sr):
    print("Listening...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    return audio.flatten()

# Load the saved SVM model
model_path = r"D:\svmtest\svm_model.joblib"
svm_model = joblib.load(model_path)
print("Model loaded successfully.")

# Parameters for audio recording
sampling_rate = 22050  # Standard sampling rate for audio
duration = 2  # Duration to record each audio clip in seconds

# Infinite loop for real-time detection
try:
    while True:
        # Capture audio from microphone
        audio = capture_audio(duration, sampling_rate)
        
        # Extract features
        features = extract_features(audio, sampling_rate)
        features = features.reshape(1, -1)  # Reshape for prediction
        
        # Predict using the SVM model
        prediction = svm_model.predict(features)[0]
        
        # Print result
        if prediction == 1:
            print("Drone detected!")
        else:
            print("Listening...")
except KeyboardInterrupt:
    print("Real-time detection stopped.")