import pyaudio
import struct
import numpy as np
import tensorflow as tf
import librosa
import wave
from scipy.io import wavfile
import IPython.display as ipd
import noisereduce as nr


model = tf.keras.models.load_model('C:\\Users\\gana4\\MAJOR PROJECT\\Speech recognition major module2.h5')

def recordAudio():
    chunk = 1024 
    sample_format = pyaudio.paInt16 
    channels = 1
    fs = 48100 
    seconds = 5
    filename = "Predict-Record-Audio.wav"

    p = pyaudio.PyAudio() 
    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = [] 
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

   
    stream.stop_stream()
    stream.close()
  
    p.terminate()

    print('Finished recording')

   
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
     
    rate, data = wavfile.read("Predict-Record-Audio.wav")
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    wavfile.write("mywav_reduced_noise.wav", rate, reduced_noise)
    
    
def extract_mfcc(wav_file_name):
    y, sr = librosa.load(wav_file_name)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,axis=0)
    return mfccs
    
wav_filepath = "C:\\Users\\gana4\\MAJOR PROJECT\\mywav_reduced_noise.wav"
def Speechrecognition():
    emotions={1 : 'neutral', 2 : 'calm', 3 : 'happy', 4 : 'sad', 5 : 'angry', 6 : 'fearful', 7 : 'disgust', 8 : 'surprised'}
    test_point=extract_mfcc(wav_filepath)
    test_point=np.reshape(test_point,newshape=(1,40,1))
    predictions=model.predict(test_point)
    varemotion = emotions[np.argmax(predictions[0] )+1]
    return varemotion