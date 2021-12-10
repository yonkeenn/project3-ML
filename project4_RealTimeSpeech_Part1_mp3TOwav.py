#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 15:31:09 2021

@author: enzocfb
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 09:22:25 2021

@author: enzocfb
"""

# las librerias requeridas
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import os
from PIL import Image
import pathlib
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import keras
from keras import layers
from keras import layers
import keras
from keras.models import Sequential
import warnings
warnings.filterwarnings('ignore')


import os
import glob
import sys
from pydub import AudioSegment   
from sklearn.metrics import roc_auc_score
# fin las librerias requeridas

def identificar_parametros(filename):
 
    cadena1 = file_name
    

    if cadena1.find("otros")>=0 :
       return 0       
    if cadena1.find("slogan1")>=0 :
       return 1
    if cadena1.find("slogan2")>=0 :
       return 2
    if cadena1.find("slogan3")>=0 :
       return 3
    if cadena1.find("slogan4")>=0 :
       return 4
    if cadena1.find("slogan5")>=0 :
       return 5
    if cadena1.find("slogan6")>=0 :
       return 6
    if cadena1.find("slogan7")>=0 :
       return 7
    if cadena1.find("slogan8")>=0 :
       return 8
    if cadena1.find("slogan9")>=0 :
       return 9
    if cadena1.find("slogan10")>=0 :
       return 10
    else:
        return 999
       

# convertir los audios en png (extraer el espectograma para cada audio)
cmap = plt.get_cmap('inferno')
plt.figure(figsize=(8,8))
# politicos = 'mazeti rafael vizcarra'.split()
slogan = 'slogan1 slogan2 slogan3 slogan3 slogan4 slogan5 slogan6 slogan7 slogan8 slogan9 otros'.split()
path = r'/home/enzocfb/py/code/ExFinal/data'

extension = '.mp3'
# extension = '.csv'


# ====================================     
# 1. Lectura de mp3, pasarlo a Wav, y Crear espectograma
#    Fase1: entrenamiento      
#    SOLO SE EJECUTA AL INICIO, UNA VEZ.
# ====================================
# for root, dirs_list, files_list in os.walk(path):
#     for file_name in files_list:
#         if os.path.splitext(file_name)[-1] == extension:
            
#             file_name_path = os.path.join(root, file_name)
            
            
#             # sound = AudioSegment.from_mp3("/path/to/file.mp3")
#             # sound.export("/output/path/file.wav", format="wav")
#             # print(audio)
#             # sound = AudioSegment.from_mp3(file_name_path)
#             # nuevo_file_wav = f'{file_name_path[:-3].replace(".", "")}.wav'            
#             # sound.export(nuevo_file_wav, format="wav")
#             # audio = nuevo_file_wav
            
           

#             y, sr = librosa.load(audio, mono=True, duration=5)
#             plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
#             plt.axis('off');
#             plt.savefig(f'{file_name_path[:-3].replace(".", "")}.png')
#             plt.clf()
