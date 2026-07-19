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

#Reproducir el audio
#playsound(r'C:\Users\PC\Documents\DocumentosGustavo\Github\Maestria\Procesamiento-Clasificacion-Datos\Procesamiento-Clasificacion-Datos\data\raw\tarea7\bisicleta_alquiler.mp3')

#Cargar el audio
audio, sr = librosa.load(r'C:\Users\PC\Documents\DocumentosGustavo\Github\Maestria\Procesamiento-Clasificacion-Datos\Procesamiento-Clasificacion-Datos\data\raw\tarea7\bisicleta_alquiler.mp3')

#===========================================================================================
#Funcion que quita el silencio del audio
#===========================================================================================
def quitar_silencio(audio):

    intervalos = librosa.effects.split(
        audio,
        top_db=25
    )

    inicio = intervalos[0][0]
    fin = intervalos[-1][1]

    return audio[inicio:fin]
#============================================================================================
#Funcion para obtener los mfcc
#============================================================================================
def obtener_mfcc(audio, sr):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )

    return mfcc
#============================================================================================
# Graficar señal-audio
#============================================================================================
plt.figure()
plt.plot(audio)
plt.title("Señal de audio original")
plt.xlabel("Tiempo(s)")
plt.ylabel("Amplitud")
# Guardar imagen en la ruta deseada
plt.savefig(
    r'C:\Users\PC\Documents\DocumentosGustavo\Github\Maestria\Procesamiento-Clasificacion-Datos\Procesamiento-Clasificacion-Datos\data\processed\tarea7\audio_plot.png',
    dpi=300, bbox_inches='tight'
)
plt.show()

#Eliminar el silencio del audio
audio = quitar_silencio(audio)

#probar si funciona el quitar silencio
plt.figure()
plt.plot(audio)
plt.title("Señal de audio sin ruido")
plt.xlabel("Tiempo(s)")
plt.ylabel("Amplitud")
# Guardar imagen en la ruta deseada
plt.savefig(
    r'C:\Users\PC\Documents\DocumentosGustavo\Github\Maestria\Procesamiento-Clasificacion-Datos\Procesamiento-Clasificacion-Datos\data\processed\tarea7\audio_sin_ruido.png',
    dpi=300, bbox_inches='tight'
)
plt.show()
#============================================================================================
# Calcular MFCC para el audio original sin ruido
#============================================================================================
mfccs_audio = obtener_mfcc(audio, sr)

plt.figure()
librosa.display.specshow(mfccs_audio, x_axis='time', sr=sr)
plt.title("Coeficientes MFCC")
plt.ylabel("Coeficiente")
plt.xlabel("Tiempo(s)")
plt.colorbar(format='%+2.0f dB')
plt.savefig(
    r'C:\Users\PC\Documents\DocumentosGustavo\Github\Maestria\Procesamiento-Clasificacion-Datos\Procesamiento-Clasificacion-Datos\data\processed\tarea7\audio_mfcc.png',
    dpi=300, bbox_inches='tight'
)
plt.show()

#===========================================================================================
# Generar audio de "bicicleta"
#===========================================================================================
tts = gTTS("bicicleta", lang="es")
tts.save(r"C:\Users\PC\Documents\DocumentosGustavo\Github\Maestria\Procesamiento-Clasificacion-Datos\Procesamiento-Clasificacion-Datos\data\raw\tarea7\bicicleta.mp3")

# Cargar el audio guardado
bic, sr_bic = librosa.load(r"C:\Users\PC\Documents\DocumentosGustavo\Github\Maestria\Procesamiento-Clasificacion-Datos\Procesamiento-Clasificacion-Datos\data\raw\tarea7\bicicleta.mp3")

plt.figure()
plt.plot(bic)
plt.title("Señal de audio: bicicleta")
plt.xlabel("Tiempo(s)")
plt.ylabel("Amplitud")
# Guardar imagen en la ruta deseada
plt.savefig(
    r'C:\Users\PC\Documents\DocumentosGustavo\Github\Maestria\Procesamiento-Clasificacion-Datos\Procesamiento-Clasificacion-Datos\data\processed\tarea7\bic.png',
    dpi=300, bbox_inches='tight'
)
plt.show()
