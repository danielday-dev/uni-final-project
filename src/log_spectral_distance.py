import librosa
import numpy as np

def log_spectral_distance(waveform1, waveform2, sr=22050, n_fft=1024, hop_length=512):
    """calculates the log-spectral distance of two waveforms
    log base e"""
    waveform1 = np.array(waveform1)
    waveform2 = np.array(waveform2)
    stft1 = librosa.stft(waveform1, n_fft=n_fft, hop_length=hop_length)
    stft2 = librosa.stft(waveform2, n_fft=n_fft, hop_length=hop_length)

    #small constant to avoid division by zero
    eps = 1e-10  
    log_spectral1 = np.log(np.abs(stft1) + eps)
    log_spectral2 = np.log(np.abs(stft2) + eps)

    #calculate the element-wise distance between the log-spectral
    dist = np.sqrt(np.mean((log_spectral1 - log_spectral2) ** 2))
    return dist
