# Universidad Autonoma de Nuevo León
## Facultad de Ciencias Físico Matematicas 
### Maestria en Ciencia de Datos

# Procesamiento y clasificación de datos.
_**Autor:**_ Gustavo de Jesús Escobar Mata.

Este es el repositorio de la materia de Datos Masivos de la Maestria de Ciencia de Datos de la Facultad de Ciencias Fisico Matematicas de la Universidad Autonoma de Nuevo León que imparte el Dr. [Alberto Benavides](https://github.com/albertobenavides) correspondiente al Cuarto Tetramestre llevado acabo de Mayo-Agosto del 2026.

# Tareas
 ## Tarea 1
  - **Objetivo** Realizar un  
  - **Solución**
 ## Tarea 2: 
  - **Objetivo:** Elegir un conjunto de datos de reseñas de usuarios y realizarles un método de vectorización adecuado ademas de estudiar sus proiedades. Finalmente realizar un análisis de sentimiento al conjunto de datos. 
  - **Solución:** Se descargaron los comentarios de los videos de Adrian de la Garza utilizando la API de Youtube en Google Cloude Platform por medio del codigo [youtube_extractor.py](Análisis-Sentimiento\extractors\youtube_extractor.py) y procesado mediante pysentimiento utilizando robertuito en el codigo [sentiment.py](Análisis-Sentimiento\processors\sentiment.py). El resultado obtenido de la extracción se encuentra en la carpeta [raw](Análisis-Sentimiento\data\raw) , mientras que en el documento [sentimiento.xlsx](Análisis-Sentimiento\data\processed) se encuentra el análisis de sentimiento obtenido. 


# Estructura del proyecto

```bash
Análisis-Sentimiento/
│
├── config/
│   └── credentials.py      
│
├── extractors/
│   ├── twitter_extractor.py
│   └── youtube_extractor.py
│
├── processors/
│   ├── cleaner.py           # Limpieza de texto
│   └── sentiment.py         # Análisis de sentimientos
│
├── data/
│   ├── raw/                 # Datos crudos
│   └── processed/           # Datos limpios
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

## Seguimiento: 
- [05/06/2026] : se probo para Twetter, pero twetter ya no permite hacer descargas de tweters gratuitos. Se evalua la opcion de pagar la suscripción de 1000 USD al mes para extraer la información. 
