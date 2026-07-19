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
import pywt

#Reproducir el audio
ruta = r"C:\Users\PC\Documents\DocumentosGustavo\Github\Maestria\Procesamiento-Clasificacion-Datos\Procesamiento-Clasificacion-Datos\data\raw\tarea7\pianog.mp3"
#playsound(ruta )

#Cargar el audio
audio, sr = librosa.load(ruta)

#============================================================================================
# Transformada Wavelet Continua (Morlet)
#============================================================================================
# Definir las escalas
escalas = np.arange(1, 128)

# Calcular la transformada wavelet
escalas = np.arange(1, 128)

#Calcular la transformada wavelet continua (CWT) usando la wavelet de Morlet
coeficientes, frecuencias = pywt.cwt(
    audio,
    escalas,
    'morl',
    sampling_period=1/sr
)

# Vector de tiempo
tiempo = np.arange(len(audio)) / sr

# Graficar el escalograma
plt.figure(figsize=(12,6))
plt.imshow(
    np.abs(coeficientes),
    extent=[tiempo[0], tiempo[-1], escalas[-1], escalas[0]],
    aspect='auto',
    cmap='jet'
)

plt.colorbar(label='Magnitud')
plt.xlabel("Tiempo (s)")
plt.ylabel("Escala")
plt.title("Transformada Wavelet Continua (Morlet)")

plt.savefig(
    r'C:\Users\PC\Documents\DocumentosGustavo\Github\Maestria\Procesamiento-Clasificacion-Datos\Procesamiento-Clasificacion-Datos\data\processed\tarea7\wavelet_morlet.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()
