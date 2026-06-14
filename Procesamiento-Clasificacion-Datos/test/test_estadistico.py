# tests/test_estadistico_fb.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from collections import Counter
import re
import unicodedata

# ─────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data", "raw")

posts_file = sorted([f for f in os.listdir(data_dir) if f.startswith("posts_samuel_fb")])[-1]
df_posts   = pd.read_csv(os.path.join(data_dir, posts_file))

com_file   = sorted([f for f in os.listdir(data_dir) if f.startswith("comentarios_samuel_fb")])[-1]
df_com     = pd.read_csv(os.path.join(data_dir, com_file))

print(f"Posts cargados      : {len(df_posts)}")
print(f"Comentarios cargados: {len(df_com)}")

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
STOPWORDS = {
    "de","la","el","en","y","a","que","los","las","es","se","del","un","una",
    "por","con","no","lo","le","su","si","pero","este","como","más","ya",
    "al","son","hay","fue","han","me","te","mi","tu","nos","para","sus"
}

def limpiar(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+", "", texto)
    texto = re.sub(r"<[^>]+>", "", texto)
    return texto.strip()

def tokenizar(texto):
    texto = limpiar(texto)
    tokens = re.findall(r'\b[a-záéíóúüñ]{3,}\b', texto)
    return [t for t in tokens if t not in STOPWORDS]

def extraer_emojis(texto):
    if not isinstance(texto, str):
        return []
    return [c for c in texto if unicodedata.category(c) == 'So']

def extraer_puntuacion(texto):
    if not isinstance(texto, str):
        return []
    return re.findall(r'[^\w\s]', texto)

def ngrams(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# ─────────────────────────────────────────
# CALCULOS
# ─────────────────────────────────────────
fuentes = [
    ("Samuel Garcia posts"   , df_posts, "text"),
    ("Ciudadanos comentarios", df_com  , "texto")
]

# --- Estadistica descriptiva ---
rows_desc = []
for nombre, df, col in fuentes:
    df = df.copy()
    df["num_palabras"]   = df[col].apply(lambda x: len(limpiar(str(x)).split()))
    df["num_caracteres"] = df[col].apply(lambda x: len(limpiar(str(x))))
    df["num_oraciones"]  = df[col].apply(
        lambda x: len(re.split(r'[.!?]+', str(x))) if isinstance(x, str) else 0
    )
    todos_tokens = []
    for texto in df[col].dropna():
        todos_tokens.extend(tokenizar(texto))

    rows_desc.append({
        "fuente"            : nombre,
        "total_textos"      : len(df),
        "total_palabras"    : int(df["num_palabras"].sum()),
        "vocabulario_unico" : len(set(todos_tokens)),
        "palabras_avg"      : round(df["num_palabras"].mean(), 2),
        "palabras_max"      : int(df["num_palabras"].max()),
        "palabras_min"      : int(df["num_palabras"].min()),
        "palabras_std"      : round(df["num_palabras"].std(), 2),
        "caracteres_avg"    : round(df["num_caracteres"].mean(), 2),
        "caracteres_max"    : int(df["num_caracteres"].max()),
        "oraciones_avg"     : round(df["num_oraciones"].mean(), 2),
    })

df_desc = pd.DataFrame(rows_desc)
print("Estadistica descriptiva calculada")

# --- Frecuencia de palabras ---
rows_freq = []
for nombre, df, col in fuentes:
    todos_tokens = []
    for texto in df[col].dropna():
        todos_tokens.extend(tokenizar(texto))
    freq = Counter(todos_tokens)
    for palabra, count in freq.most_common(50):
        rows_freq.append({
            "fuente"    : nombre,
            "palabra"   : palabra,
            "frecuencia": count
        })

df_freq = pd.DataFrame(rows_freq)
print("Frecuencia de palabras calculada")

# --- N-gramas ---
rows_ng = []
for nombre, df, col in fuentes:
    todos_tokens = []
    for texto in df[col].dropna():
        todos_tokens.extend(tokenizar(texto))
    for n, tipo in [(2, "bigrama"), (3, "trigrama")]:
        ng   = ngrams(todos_tokens, n)
        freq = Counter(ng)
        for gram, count in freq.most_common(20):
            rows_ng.append({
                "fuente"    : nombre,
                "tipo"      : tipo,
                "ngrama"    : gram,
                "frecuencia": count
            })

df_ng = pd.DataFrame(rows_ng)
print("N-gramas calculados")

# --- Signos de puntuacion ---
rows_pun = []
for nombre, df, col in fuentes:
    toda_pun = []
    for texto in df[col].dropna():
        toda_pun.extend(extraer_puntuacion(str(texto)))
    freq = Counter(toda_pun)
    for signo, count in freq.most_common(20):
        rows_pun.append({
            "fuente"    : nombre,
            "signo"     : signo,
            "frecuencia": count
        })

df_pun = pd.DataFrame(rows_pun)
print("Puntuacion calculada")

# --- Emojis ---
rows_emoji = []
for nombre, df, col in fuentes:
    todos_emojis    = []
    textos_con_emoji = 0
    for texto in df[col].dropna():
        emojis = extraer_emojis(str(texto))
        todos_emojis.extend(emojis)
        if emojis:
            textos_con_emoji += 1
    freq = Counter(todos_emojis)
    for emoji, count in freq.most_common():
        rows_emoji.append({
            "fuente"             : nombre,
            "emoji"              : emoji,
            "frecuencia"         : count,
            "textos_con_emoji"   : textos_con_emoji,
            "total_emojis"       : len(todos_emojis),
            "emojis_unicos"      : len(freq)
        })

df_emoji = pd.DataFrame(rows_emoji)
print("Emojis calculados")

# ─────────────────────────────────────────
# GUARDAR EN EXCEL CON MULTIPLES HOJAS
# ─────────────────────────────────────────
output_dir = os.path.join(base_dir, "data", "processed")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "estadistico_fb.xlsx")

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    df_desc.to_excel(writer,  sheet_name="1_Descriptiva",   index=False)
    df_freq.to_excel(writer,  sheet_name="2_Frecuencias",   index=False)
    df_ng.to_excel(writer,    sheet_name="3_NGramas",       index=False)
    df_pun.to_excel(writer,   sheet_name="4_Puntuacion",    index=False)
    df_emoji.to_excel(writer, sheet_name="5_Emojis",        index=False)

print(f"\nArchivo guardado en: {output_path}")
print("\nHojas generadas:")
print("  1_Descriptiva  -> estadistica basica por fuente")
print("  2_Frecuencias  -> top 50 palabras por fuente")
print("  3_NGramas      -> bigramas y trigramas por fuente")
print("  4_Puntuacion   -> signos de puntuacion por fuente")
print("  5_Emojis       -> emojis usados por fuente")