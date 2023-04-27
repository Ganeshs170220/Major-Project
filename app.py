import pyttsx3
#import numpy as np
#import nltk
#import random
#import json
#import torch
#import sys
#import torch.nn as nn
from flask import Flask, render_template
#from torch.utils.data import Dataset, DataLoader
#from nltk.stem.porter import PorterStemmer
import speech_recognition as sr
#user defined modules
import Face_exp_model
import speech_recognition_model
#from model import NeuralNet

app = Flask(__name__)
app.static_folder = 'static'
# Set up the speech recognition object
r = sr.Recognizer()

# Set up the microphone stream
mic = sr.Microphone()

# Set up the text-to-speech engine
engine = pyttsx3.init()

x = None
y = None
@app.route('/', methods=['GET'])
def home():
        return render_template("index.html")
   
@app.route('/speech', methods=['GET'])
def run_ml_code_speech():
    global y 
     
    speech_recognition_model.recordAudio()
    speech_output = speech_recognition_model.Speechrecognition()
    y = speech_output
    return y


@app.route('/cam', methods=['GET'])
def run_ml_code():
    global x
    face_output = Face_exp_model.Facerecognization()
    x = face_output
    return x



if __name__ == '__main__':
    app.run(debug=True)
