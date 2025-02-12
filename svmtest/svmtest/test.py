import numpy as np
import librosa
import os
import joblib
import datetime

# Function to extract features from audio files
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# Function to predict for new test files
def predict_test_data(test_folder, model):
    test_features = []
    test_files = []

    for file in os.listdir(test_folder):
        if file.endswith('.wav'):
            file_path = os.path.join(test_folder, file)
            features = extract_features(file_path)
            test_features.append(features)
            test_files.append(file)

    test_features = np.array(test_features)
    predictions = model.predict(test_features)

    # Print predictions
    for i, file in enumerate(test_files):
        print(f"File: {file}, Predicted Class: {predictions[i]}")

# Load the saved model
model_path = r"D:\svmtest\svm_model.joblib"     
svm_model = joblib.load(model_path)
print("Model loaded successfully.")

# Path to test data folder
test_folder = r'D:\svmtest\testdata'

# Predict and display results

predict_test_data(test_folder, svm_model)
