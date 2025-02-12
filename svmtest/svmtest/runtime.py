import numpy as np
import os
import wave

# Function to read WAV files without scipy
def read_wav(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        raw_data = wav_file.readframes(n_frames)
        dtype = np.int16 if sample_width == 2 else np.int8
        audio_data = np.frombuffer(raw_data, dtype=dtype)
        # Ensure mono channel
        if n_channels > 1:
            audio_data = audio_data[::n_channels]
        return frame_rate, audio_data

# Function to extract basic audio features: mean, std, and zero-crossing rate
def extract_features(audio_data, sample_rate):
    mean = np.mean(audio_data)
    std = np.std(audio_data)
    zcr = np.sum(audio_data[:-1] * audio_data[1:] < 0) / len(audio_data)

    # Additional feature for silence detection: energy
    energy = np.sum(audio_data ** 2) / len(audio_data)

    return np.array([mean, std, zcr, energy])

# Load dataset
def load_audio_data(data_dir):
    features = []
    labels = []

    for label_dir in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_dir)
        if os.path.isdir(label_path):
            for file_name in os.listdir(label_path):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(label_path, file_name)
                    sample_rate, audio_data = read_wav(file_path)

                    # Extract features
                    features.append(extract_features(audio_data, sample_rate))
                    labels.append(label_dir)

    return np.array(features), np.array(labels)

# Directory structure:
# data/
#   background_noise/
#     audio1.wav
#     audio2.wav
#   drone_sound/
#     audio1.wav
#     audio2.wav

data_dir = "D:\hope\data"  # Replace with your dataset path

# Load features and labels
X, y = load_audio_data(data_dir)

# Encode labels to binary format
label_map = {label: idx for idx, label in enumerate(np.unique(y))}
y = np.array([label_map[label] for label in y])

# Split data
def train_test_split(X, y, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_idx = int(len(X) * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def train_svm(X, y, C=1.0, tol=1e-3, max_iter=1000):
    n_samples, n_features = X.shape
    alpha = np.zeros(n_samples)
    b = 0
    K = np.array([[linear_kernel(X[i], X[j]) for j in range(n_samples)] for i in range(n_samples)])

    for _ in range(max_iter):
        for i in range(n_samples):
            condition = y[i] * (np.sum(alpha * y * K[:, i]) + b) < 1
            gradient = 1 - y[i] * (np.sum(alpha * y * K[:, i]) + b)
            if condition:
                alpha[i] += C * gradient

        alpha = np.clip(alpha, 0, C)

    w = np.sum((alpha * y)[:, None] * X, axis=0)
    b = np.mean(y - np.dot(X, w))
    return w, b

def predict_svm(X, w, b):
    return np.sign(np.dot(X, w) + b)

w, b = train_svm(X_train, y_train)
y_pred = predict_svm(X_test, w, b)

# Evaluate model
def classification_report(y_true, y_pred, target_names):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    accuracy = (tp + tn) / len(y_true)

    return f"\nClassification Report:\n\n" + \
           f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}\n"

print(classification_report(y_test, y_pred, target_names=list(label_map.keys())))

# # Classify a new input file
def classify_input(file_path, w, b):
    sample_rate, audio_data = read_wav(file_path)
    features = extract_features(audio_data, sample_rate).reshape(1, -1)
    prediction = predict_svm(features, w, b)
    label = list(label_map.keys())[list(label_map.values()).index(int(prediction))]
    return label

# Classify a new input file
def classify_input(file_path, w, b):
    sample_rate, audio_data = read_wav(file_path)
    features = extract_features(audio_data, sample_rate).reshape(1, -1)
    prediction = predict_svm(features, w, b)
    
    # Ensure prediction is a single value
    predicted_label_index = int(prediction[0])
    label = list(label_map.keys())[list(label_map.values()).index(predicted_label_index)]
    
    return label

# Example usage of classify_input
input_file = "D:\DroneAudioDataset-master\Binary_Drone_Audio\yes_drone\B_S2_D1_069-bebop_001_.wav"
predicted_label = classify_input(input_file, w, b)
print(f"The predicted label for the input file is: {predicted_label}")