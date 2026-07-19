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
#playsound(r'C:\Users\PC\Documents\DocumentosGustavo\Github\Maestria\Procesamiento-Clasificacion-Datos\Procesamiento-Clasificacion-Datos\data\raw\tarea6\bisicleta_alquiler.mp3')

#Cargar el audio
audio, sr = librosa.load(r'C:\Users\PC\Documents\DocumentosGustavo\Github\Maestria\Procesamiento-Clasificacion-Datos\Procesamiento-Clasificacion-Datos\data\raw\tarea6\bisicleta_alquiler.mp3')

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
#Funcion para comparar dos audios y obtener la distancia euclidiana entre sus mfcc
#============================================================================================
def distancia(mfcc1, mfcc2):

    alignment = dtw(
        mfcc1.T,
        mfcc2.T,
        keep_internals=True
    )

    return alignment.distance
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
    r'C:\Users\PC\Documents\DocumentosGustavo\Github\Maestria\Procesamiento-Clasificacion-Datos\Procesamiento-Clasificacion-Datos\data\processed\tarea6\audio_plot.png',
    dpi=300, bbox_inches='tight'
)
#plt.show()

#Eliminar el silencio del audio
audio = quitar_silencio(audio)

#probar si funciona el quitar silencio
plt.figure()
plt.plot(audio)
plt.title("Señal de audio original sin ruido")
plt.xlabel("Tiempo(s)")
plt.ylabel("Amplitud")
# Guardar imagen en la ruta deseada
plt.savefig(
    r'C:\Users\PC\Documents\DocumentosGustavo\Github\Maestria\Procesamiento-Clasificacion-Datos\Procesamiento-Clasificacion-Datos\data\processed\tarea6\audio_sin_ruido.png',
    dpi=300, bbox_inches='tight'
)
#plt.show()
#============================================================================================
# Calcular MFCC para el audio original sin ruido
#============================================================================================
mfccs_audio = obtener_mfcc(audio, sr)

plt.figure()
librosa.display.specshow(mfccs_audio, x_axis='time', sr=sr)
plt.title("Coeficientes MFCC audio original")
plt.ylabel("Coeficiente")
plt.xlabel("Tiempo(s)")
plt.colorbar(format='%+2.0f dB')
plt.savefig(
    r'C:\Users\PC\Documents\DocumentosGustavo\Github\Maestria\Procesamiento-Clasificacion-Datos\Procesamiento-Clasificacion-Datos\data\processed\tarea6\audio_mfcc.png',
    dpi=300, bbox_inches='tight'
)
#plt.show()

#===========================================================================================
# Generar audio de "bicicleta"
#===========================================================================================
tts = gTTS("bicicleta", lang="es")
tts.save(r"C:\Users\PC\Documents\DocumentosGustavo\Github\Maestria\Procesamiento-Clasificacion-Datos\Procesamiento-Clasificacion-Datos\data\raw\tarea6\bicicleta.mp3")

# Cargar el audio guardado
bic, sr_bic = librosa.load(r"C:\Users\PC\Documents\DocumentosGustavo\Github\Maestria\Procesamiento-Clasificacion-Datos\Procesamiento-Clasificacion-Datos\data\raw\tarea6\bicicleta.mp3")

plt.figure()
plt.plot(bic)
plt.title("Señal de audio: bicicleta")
plt.xlabel("Tiempo(s)")
plt.ylabel("Amplitud")
# Guardar imagen en la ruta deseada
plt.savefig(
    r'C:\Users\PC\Documents\DocumentosGustavo\Github\Maestria\Procesamiento-Clasificacion-Datos\Procesamiento-Clasificacion-Datos\data\processed\tarea6\bic.png',
    dpi=300, bbox_inches='tight'
)
#plt.show()

#Eliminar el silencio del audio
bic = quitar_silencio(bic)

#probar si funciona el quitar silencio
plt.figure()
plt.plot(bic)
plt.title("Señal de bicicleta sin ruido")
plt.xlabel("Tiempo(s)")
plt.ylabel("Amplitud")
# Guardar imagen en la ruta deseada
plt.savefig(
    r'C:\Users\PC\Documents\DocumentosGustavo\Github\Maestria\Procesamiento-Clasificacion-Datos\Procesamiento-Clasificacion-Datos\data\processed\tarea6\bic_sin_ruido.png',
    dpi=300, bbox_inches='tight'
)
#plt.show()

#============================================================================================
# Calcular MFCC para el audio bic sin ruido
#============================================================================================
mfccs_bic = obtener_mfcc(bic, sr_bic)

plt.figure()
librosa.display.specshow(mfccs_bic, x_axis='time', sr=sr_bic)
plt.title("Coeficientes MFCC bicicleta")
plt.ylabel("Coeficiente")
plt.xlabel("Tiempo(s)")
plt.colorbar(format='%+2.0f dB')
plt.savefig(
    r'C:\Users\PC\Documents\DocumentosGustavo\Github\Maestria\Procesamiento-Clasificacion-Datos\Procesamiento-Clasificacion-Datos\data\processed\tarea6\bic_mfcc.png',
    dpi=300, bbox_inches='tight'
)
#plt.show()

#===========================================================================================
# Generar audio de "alquiler"
#===========================================================================================
tts = gTTS("alquiler", lang="es")
tts.save(r"C:\Users\PC\Documents\DocumentosGustavo\Github\Maestria\Procesamiento-Clasificacion-Datos\Procesamiento-Clasificacion-Datos\data\raw\tarea6\alquiler.mp3")

# Cargar el audio guardado
alq, sr_alq = librosa.load(r"C:\Users\PC\Documents\DocumentosGustavo\Github\Maestria\Procesamiento-Clasificacion-Datos\Procesamiento-Clasificacion-Datos\data\raw\tarea6\alquiler.mp3")

plt.figure()
plt.plot(alq)
plt.title("Señal de audio: alquiler")
plt.xlabel("Tiempo(s)")
plt.ylabel("Amplitud")
# Guardar imagen en la ruta deseada
plt.savefig(
    r'C:\Users\PC\Documents\DocumentosGustavo\Github\Maestria\Procesamiento-Clasificacion-Datos\Procesamiento-Clasificacion-Datos\data\processed\tarea6\alq.png',
    dpi=300, bbox_inches='tight'
)
#plt.show()

#Eliminar el silencio del audio
alq = quitar_silencio(alq)

#probar si funciona el quitar silencio
plt.figure()
plt.plot(alq)
plt.title("Señal de audio alquiler sin ruido")
plt.xlabel("Tiempo(s)")
plt.ylabel("Amplitud")
# Guardar imagen en la ruta deseada
plt.savefig(
    r'C:\Users\PC\Documents\DocumentosGustavo\Github\Maestria\Procesamiento-Clasificacion-Datos\Procesamiento-Clasificacion-Datos\data\processed\tarea6\alq_sin_ruido.png',
    dpi=300, bbox_inches='tight'
)
#plt.show()

#============================================================================================
# Calcular MFCC para el audio alq sin ruido
#============================================================================================
mfccs_alq = obtener_mfcc(alq, sr_alq)

plt.figure()
librosa.display.specshow(mfccs_alq, x_axis='time', sr=sr_alq)
plt.title("Coeficientes MFCC alquiler")
plt.ylabel("Coeficiente")
plt.xlabel("Tiempo(s)")
plt.colorbar(format='%+2.0f dB')
plt.savefig(
    r'C:\Users\PC\Documents\DocumentosGustavo\Github\Maestria\Procesamiento-Clasificacion-Datos\Procesamiento-Clasificacion-Datos\data\processed\tarea6\alq_mfcc.png',
    dpi=300, bbox_inches='tight'
)
#plt.show()


#===========================================================================================
# Comparar los audios y obtener la distancia euclidiana entre sus mfcc
#===========================================================================================
# Distancia entre el audio de prueba y "bicicleta"
distancia_bic = distancia(mfccs_audio, mfccs_bic)

# Distancia entre el audio de prueba y "alquiler"
distancia_alq = distancia(mfccs_audio, mfccs_alq)

print("="*60)
print("Distancias obtenidas")
print("="*60)
print(f"Distancia con 'bicicleta' : {distancia_bic:.2f}")
print(f"Distancia con 'alquiler'  : {distancia_alq:.2f}")
print("="*60)

# Decisión
if distancia_bic < distancia_alq:
    print("RESULTADO: El audio se parece más a 'bicicleta'")
else:
    print("RESULTADO: El audio se parece más a 'alquiler'")

# Diferencia entre ambas distancias
diferencia = abs(distancia_bic - distancia_alq)

print(f"\nDiferencia entre distancias: {diferencia:.2f}")
