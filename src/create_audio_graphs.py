import matplotlib.pyplot as plt
import numpy as np
import wave
import sys

def plot_time_domain(signal, sample_rate):
    '''plots time domain graph for audio signal'''
    time = np.arange(0, len(signal)) / sample_rate
    plt.figure(figsize=(10, 5))
    plt.plot(time, signal)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Time Domain')
    plt.grid(True)
    plt.show()

def plot_frequency_domain(signal, sample_rate):
    '''plots frequency domain'''
    n = len(signal)
    fft_result = np.fft.fft(signal)
    freq = np.fft.fftfreq(n, d=1/sample_rate)
    magnitude = np.abs(fft_result)

    plt.figure(figsize=(10, 5))
    plt.plot(freq[:n//2], magnitude[:n//2])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Frequency Domain')
    plt.grid(True)
    plt.show()

def main():
    wav_file = "./final_out/audio_test_1.wav"

    # Open the WAV file
    try:
        with wave.open(wav_file, 'rb') as wf:
            num_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            num_frames = wf.getnframes()
            raw_data = wf.readframes(num_frames)
    except FileNotFoundError:
        print(f"Error: File '{wav_file}' not found.")
        sys.exit(1)

    print(num_channels, sample_width, sample_rate, num_frames)

    # Convert the raw data to numpy array based on sample width and number of channels
    if sample_width == 1:
        dtype = np.uint8
    elif sample_width == 4:
        dtype = np.int16
    else:
        print("Error: Unsupported sample width.")
        sys.exit(1)

    signal = np.frombuffer(raw_data, dtype=dtype)

    # If stereo, take only one channel
    if num_channels == 2:
        signal = signal[::2]

    # Normalize the signal (optional)
    signal = signal / np.max(np.abs(signal))

    # Plot the time and frequency domain
    plot_time_domain(signal, sample_rate)
    plot_frequency_domain(signal, sample_rate)

if __name__ == "__main__":
    main()