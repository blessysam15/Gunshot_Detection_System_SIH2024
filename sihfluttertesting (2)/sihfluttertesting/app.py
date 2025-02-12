from flask import Flask, jsonify, render_template
import tensorflow as tf
import numpy as np
import sounddevice as sd
import librosa
import wavio
import os
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Create uploads directory if it doesn't exist
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Load TFLite model
tflite_model_path = os.path.join("model", "newmodel.tflite")

def record_audio(filename, duration=5, sample_rate=48000):
    """Record audio and save it to a file."""
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype='float32')
    sd.wait()
    wavio.write(filename, audio, sample_rate, sampwidth=3)

def record_audio(filename, duration=5, sample_rate=48000):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype='float32')
    sd.wait()
    wavio.write(filename, audio, sample_rate, sampwidth=3) # Save as .wav file
    print(f"Audio saved as {filename}")

def load_audio(file_path, tsr=48000):
    audio, sample_rate = librosa.load(file_path, sr=tsr)  # Resamples to target sample rate
    print(f"Loaded audio sample rate: {sample_rate}")
    return audio

def preprocess_audio(audio, target_shape=(1, 70000)):
    audio = np.clip(audio, -1, 1)
    if len(audio) < target_shape[1]:
        audio = np.pad(audio, (0, target_shape[1] - len(audio)), mode='constant')
    else:
        audio = audio[:target_shape[1]]
    return np.expand_dims(audio, axis=0).astype(np.float32)

def classify_audio(tflite_model_path, audio_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # print(input_details)
    # print(output_details)

    audio = load_audio(audio_path)
    input_data = preprocess_audio(audio, target_shape=input_details[0]['shape'])

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(output_data)
    return predicted_label, output_data

@app.route('/')
def index():
    """Render the HTML page."""
    return render_template("index.html")

@app.route('/detection')
def detection():
    try:
        # Record audio and save it
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"uploads/audio_{timestamp}.wav"
        record_audio(audio_filename, duration=5)

        # Classify recorded audio
        predicted_label, confidence_scores = classify_audio(tflite_model_path, audio_filename)
        print("predicted label:", predicted_label)
        # Convert confidence_scores to a list for JSON serialization
        response = {
            "timestamp": timestamp,
            "predicted_label": str(predicted_label),
            "confidence_scores": confidence_scores.tolist()  # Properly serialize ndarray
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
