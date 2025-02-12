import numpy as np
import librosa
import sounddevice as sd
import joblib

# Function to extract features from audio
def extract_features_from_audio(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# Function to capture audio from the microphone
def capture_audio(duration, sr):
    print("Listening...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    return audio.flatten()

# Load the saved SVM model
model_path = r"D:\svmtest\svm_modelwithoutlibrosa.joblib"
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
        features = extract_features_from_audio(audio, sampling_rate)
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
