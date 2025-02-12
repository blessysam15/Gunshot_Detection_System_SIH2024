import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Function to extract features from audio files
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# Paths for audio datasets
class_0_path = r"D:\svmtest\class0"  # Folder for class 0 audio files
class_1_path = r"D:\svmtest\class1"  # Folder for class 1 audio files

# Data and Labels
X, y = [], []

# Extract features for class 0
for file in os.listdir(class_0_path):
    if file.endswith('.wav'):
        X.append(extract_features(os.path.join(class_0_path, file)))
        y.append(0)

# Extract features for class 1
for file in os.listdir(class_1_path):
    if file.endswith('.wav'):
        X.append(extract_features(os.path.join(class_1_path, file)))
        y.append(1)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train, y_train)

# Evaluate the model
y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
model_path = r"D:\svmtest\svm_model1.joblib"
joblib.dump(svm_model, model_path)
print(f"Model saved to {model_path}")
