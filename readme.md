# Universidad Autonoma de Nuevo León
## Facultad de Ciencias Físico Matematicas 
### Maestria en Ciencia de Datos

# Procesamiento y clasificación de datos.
_**Autor:**_ Gustavo de Jesús Escobar Mata.

Este es el repositorio de la materia de Datos Masivos de la Maestria de Ciencia de Datos de la Facultad de Ciencias Fisico Matematicas de la Universidad Autonoma de Nuevo León que imparte el Dr. [Alberto Benavides](https://github.com/albertobenavides) correspondiente al Cuarto Tetramestre llevado acabo de Mayo-Agosto del 2026.

# Tareas
 ## Tarea 1
  - **Objetivo:** Realizar un análisis estadistico sobre algún origen de datos textual (libros, publicaciones en redes sociales, entradas de blog, discursos politicos,...) sobre dos o mas fuentes de uno o mas autores. Análizar estadistica descriptiva basica, frcuencias, discuros de palabras , n-gramas, usos de signo de puntuación, emojis, etc. 
  - **Solución:** Se extrajo los comentarios de 20 post en Facebook del Gobernador Samuel Garcia con 50 comentarios cada uno utilizando el codigo [facebook_extractor.py](Procesamiento-Clasificacion-Datos\extractors\facebook_extractor.py). El resultado se le implementaron [pruebas estadisticas](Procesamiento-Clasificacion-Datos\data\processed\estadistico_fb.xlsx) para determinar las caracteristicas de los datos. 
 ## Tarea 2: 
  - **Objetivo:** Elegir un conjunto de datos de reseñas de usuarios y realizarles un método de vectorización adecuado ademas de estudiar sus proiedades. Finalmente realizar un análisis de sentimiento al conjunto de datos. 
  - **Solución:** Se descargaron los comentarios de los videos de Adrian de la Garza utilizando la API de Youtube en Google Cloude Platform por medio del codigo [youtube_extractor.py](Procesamiento-Clasificacion-Datos\extractors\youtube_extractor.py) y procesado mediante pysentimiento utilizando robertuito en el codigo [sentiment.py](Procesamiento-Clasificacion-Datos\processors\sentiment.py). El resultado obtenido de la extracción se encuentra en la carpeta [raw](Procesamiento-Clasificacion-Datos\data\raw) , mientras que en el documento [sentimiento.xlsx](Procesamiento-Clasificacion-Datos\data\processed) se encuentra el análisis de sentimiento obtenido. 
## Tarea 3:
  - **Objetivo:** Hacer un diseño de experimentos para comparar modelos y sus hiperparametros con relación a la clasificación de textos. 
  - **Solución:** Se realizo un análisis de experimentos con los modelos tranformers robertuito (base), XLM-RoBERTa, DistilBERT-Multi, MarlA-ROBERTa y modelos clasicos como Naive Bayes y SVM Lineal C=1 y C=0. El mejor modelo, comparando como robertuito, es XLM-RoBERTa teniendo una metrica de 0.7534 de f1 tardando un tiempo total de 283.24 segundos. Los demas resultados se encuentran en el documento [experimento_clasificacion.xlsx](Procesamiento-Clasificacion-Datos\data\processed\experimento_clasificacion.xlsx)
  


# Estructura del proyecto

```bash
Procesamiento-Clasificacion-Datos/
│
├── config/
│   └── credentials.py      
│
├── data/
│   ├── raw/                 # Datos crudos
│   └── processed/           # Datos limpios
│
├── experiments/
│   └── experiment.py        #Experimentos
│   
├── extractors/
│   ├── facebook_extractor.py
│   ├── twitter_extractor.py
│   └── youtube_extractor.py
│
├── processors/
│   ├── cleaner.py           # Limpieza de texto
│   └── sentiment.py         # Análisis de sentimientos
│
├── test/
│   ├── test_estadistico.py     
│   ├── test_experimento.py     
│   ├── test_facebook.py        
│   ├── test_sentimient_fb.py   
│   ├── test_sentimient.py      
│   ├── test_twitter.py         
│   └── test_youtube.py         
│
│
└── main.py                  # Orquestador del pipeline
```

## Objetivo
- Agenda setting - ¿Que temas impulsa antes de eventos políticos?
- Sentiment shift — ¿Cómo cambia la percepción ciudadana en el tiempo?
- Bots/astroturfing — Cuentas con comportamiento anómalo en sus replies
- Red de aliados — Grafo de menciones frecuentes (NetworkX)
- Discurso vs realidad — Sus palabras clave vs noticias de ese periodo

## Seguimiento 
- [05/06/2026] : se probo para Twetter, pero twetter ya no permite hacer descargas de tweters gratuitos. Se evalua la opcion de pagar la suscripción de 1000 USD al mes para extraer la información. 

# Programas
APIfy https://console.apify.com/
Google Cloud Platform: Youtube API 
X for Developers
