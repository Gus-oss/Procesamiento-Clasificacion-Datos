#audio.py
#===========================================================================================
#Librerias
#===========================================================================================
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import IPython
from playsound import playsound
from gtts import gTTS #Generar audio a partir de texto
from dtw import dtw #compara las distancias euclidianas

#Reproducir el audio
playsound(r'C:\Users\PC\Documents\DocumentosGustavo\Github\Maestria\Procesamiento-Clasificacion-Datos\Procesamiento-Clasificacion-Datos\data\raw\tarea7\pianog.mp3')

#Cargar el audio
audio, sr = librosa.load(r'C:\Users\PC\Documents\DocumentosGustavo\Github\Maestria\Procesamiento-Clasificacion-Datos\Procesamiento-Clasificacion-Datos\data\raw\tarea7\pianog.mp3')
