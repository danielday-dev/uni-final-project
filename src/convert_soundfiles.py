import os
import subprocess

# input and output directories

input_dir = "./raw_audio"
hq_output_dir = "./audio_hq"
lq_output_dir = "./audio_lq"

lq_sample_rate = "48000"
hq_sample_rate = "8000"

for filename in os.listdir(input_dir):

    # create the output filename by replacing the .flac extension with .wav
    output_filename = os.path.splitext(filename)[0] + ".wav"
    #build the full path to the input and output files
    input_path = os.path.join(input_dir, filename)
    hq_output_path = os.path.join(hq_output_dir, output_filename)
    # print(hq_output_path)
    lq_output_path = os.path.join(lq_output_dir, output_filename)
    
    #use ffmpeg command-line tool to convert file with specified parameters
    if not os.path.isfile(hq_output_path): 
        subprocess.run(["C:/Users/danie/Downloads/ffmpeg/bin/ffmpeg.exe", "-i", input_path, "-ar", lq_sample_rate, "-ac", "1", hq_output_path])
    if not os.path.isfile(lq_output_path): 
        subprocess.run(["C:/Users/danie/Downloads/ffmpeg/bin/ffmpeg.exe", "-i", input_path, "-ar", hq_sample_rate, "-ac", "1", lq_output_path])
    
