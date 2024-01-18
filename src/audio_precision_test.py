import torch
import torchaudio
import os

#program to test different levels of rounding on audio files

TESTAUDIO = "./audio_hq/p225_001_mic1.wav"
OUTPATH = "./out/p225_001_mic1.wav"
DECIMALS = 2

wave = torchaudio.load(TESTAUDIO)[0]
    
print(wave)
print(wave.size(1))
# for i in range(wave.size(1)):
#     wave[i] = round(wave[i], 3)

#adjust level of rounding
wave = wave.round(decimals = DECIMALS)
print(wave)

torchaudio.save(OUTPATH, wave, 48000, encoding = "PCM_S")