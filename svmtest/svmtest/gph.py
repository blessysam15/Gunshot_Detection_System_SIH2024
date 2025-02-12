import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Serial port configurations
master_port = serial.Serial("COM10", 1000000, timeout=1)
slave_port = serial.Serial("COM13", 1000000, timeout=1)

# Parameters
SAMPLE_RATE = 48000
WINDOW_SIZE = 1024
MAX_DELAY = 0.00035  # Max delay for TDOA, adjust based on mic distance and speed of sound
epsilon = 1e-10  # Small value to prevent division by zero in GCC-PHAT

# Real-time TDOA plot
fig, ax = plt.subplots()
ax.set_title("Real-Time TDOA")
ax.set_xlabel("Time (frames)")
ax.set_ylabel("TDOA (s)")
line, = ax.plot([], [], lw=2)
x_data, y_data = [], []

def init():
    ax.set_xlim(0, 100)
    ax.set_ylim(-MAX_DELAY, MAX_DELAY)
    return line,

def update(frame):
    # Read audio data from both mics
    master_data = np.frombuffer(master_port.read(WINDOW_SIZE * 2), dtype=np.int16)
    slave_data = np.frombuffer(slave_port.read(WINDOW_SIZE * 2), dtype=np.int16)
    
    # Calculate TDOA using GCC-PHAT
    tdoa = gcc_phat(master_data, slave_data, SAMPLE_RATE)
    
    # Update plot data
    x_data.append(len(x_data))
    y_data.append(tdoa)
    
    if len(x_data) > 100:  # Keep a fixed-length window
        x_data.pop(0)
        y_data.pop(0)
    
    line.set_data(x_data, y_data)
    return line,

def gcc_phat(sig, refsig, fs):
    """Generalized Cross-Correlation with Phase Transform (GCC-PHAT)"""
    n = len(sig) + len(refsig)
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    cross_spectrum = SIG * np.conj(REFSIG)
    cross_spectrum /= (np.abs(cross_spectrum) + epsilon)  # Avoid divide by zero
    
    cross_corr = np.fft.irfft(cross_spectrum, n=n)
    max_shift = int(fs * MAX_DELAY)
    cross_corr = np.concatenate((cross_corr[-max_shift:], cross_corr[:max_shift + 1]))
    
    shift = np.argmax(np.abs(cross_corr)) - max_shift
    tdoa = shift / fs
    return tdoa

ani = FuncAnimation(fig, update, init_func=init, blit=True, interval=50, cache_frame_data=False)

print("Processing real-time audio data and plotting TDOA...")
plt.show()
