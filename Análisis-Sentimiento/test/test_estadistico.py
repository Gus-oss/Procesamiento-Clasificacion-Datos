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

# Posts de Samuel Garcia (autor)
posts_file = sorted([f for f in os.listdir(data_dir) if f.startswith("posts_samuel_fb")])[-1]
df_posts   = pd.read_csv(os.path.join(data_dir, posts_file))

# Comentarios de ciudadanos (audiencia)
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
    texto = re.sub(r"<[^>]+>", "", texto)          # HTML tags
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
# 1. ESTADISTICA DESCRIPTIVA BASICA
# ─────────────────────────────────────────
print("\n" + "="*60)
print("1. ESTADISTICA DESCRIPTIVA BASICA")
print("="*60)

for nombre, df, col_texto in [
    ("SAMUEL GARCIA (posts)"   , df_posts, "text"),
    ("CIUDADANOS (comentarios)", df_com  , "texto")
]:
    df = df.copy()
    df["texto_limpio"]   = df[col_texto].apply(limpiar)
    df["num_palabras"]   = df["texto_limpio"].apply(lambda x: len(x.split()))
    df["num_caracteres"] = df["texto_limpio"].apply(len)
    df["num_oraciones"]  = df[col_texto].apply(
        lambda x: len(re.split(r'[.!?]+', str(x))) if isinstance(x, str) else 0
    )

    print(f"\n--- {nombre} ---")
    print(f"  Total textos         : {len(df)}")
    print(f"  Palabras por texto   : avg={df['num_palabras'].mean():.1f}  "
          f"min={df['num_palabras'].min()}  max={df['num_palabras'].max()}  "
          f"std={df['num_palabras'].std():.1f}")
    print(f"  Caracteres por texto : avg={df['num_caracteres'].mean():.1f}  "
          f"min={df['num_caracteres'].min()}  max={df['num_caracteres'].max()}")
    print(f"  Oraciones por texto  : avg={df['num_oraciones'].mean():.1f}")
    print(f"  Total palabras       : {df['num_palabras'].sum()}")

# ─────────────────────────────────────────
# 2. FRECUENCIA DE PALABRAS
# ─────────────────────────────────────────
print("\n" + "="*60)
print("2. FRECUENCIA DE PALABRAS (top 20, sin stopwords)")
print("="*60)

for nombre, df, col_texto in [
    ("SAMUEL GARCIA (posts)"   , df_posts, "text"),
    ("CIUDADANOS (comentarios)", df_com  , "texto")
]:
    todos_tokens = []
    for texto in df[col_texto].dropna():
        todos_tokens.extend(tokenizar(texto))

    freq = Counter(todos_tokens)
    print(f"\n--- {nombre} ---")
    print(f"  Vocabulario unico: {len(freq)} palabras")
    print(f"  Top 20 palabras:")
    for palabra, count in freq.most_common(20):
        barra = "#" * int(count / max(1, freq.most_common(1)[0][1]) * 20)
        print(f"    {palabra:<20} {count:>5}  {barra}")

# ─────────────────────────────────────────
# 3. N-GRAMAS
# ─────────────────────────────────────────
print("\n" + "="*60)
print("3. N-GRAMAS MAS FRECUENTES")
print("="*60)

for nombre, df, col_texto in [
    ("SAMUEL GARCIA (posts)"   , df_posts, "text"),
    ("CIUDADANOS (comentarios)", df_com  , "texto")
]:
    todos_tokens = []
    for texto in df[col_texto].dropna():
        todos_tokens.extend(tokenizar(texto))

    print(f"\n--- {nombre} ---")
    for n, etiqueta in [(2, "Bigramas"), (3, "Trigramas")]:
        ng    = ngrams(todos_tokens, n)
        freq  = Counter(ng)
        print(f"\n  {etiqueta} (top 10):")
        for gram, count in freq.most_common(10):
            print(f"    {gram:<35} {count}")

# ─────────────────────────────────────────
# 4. USO DE SIGNOS DE PUNTUACION
# ─────────────────────────────────────────
print("\n" + "="*60)
print("4. USO DE SIGNOS DE PUNTUACION")
print("="*60)

for nombre, df, col_texto in [
    ("SAMUEL GARCIA (posts)"   , df_posts, "text"),
    ("CIUDADANOS (comentarios)", df_com  , "texto")
]:
    toda_puntuacion = []
    for texto in df[col_texto].dropna():
        toda_puntuacion.extend(extraer_puntuacion(texto))

    freq = Counter(toda_puntuacion)
    print(f"\n--- {nombre} ---")
    print(f"  Total signos: {len(toda_puntuacion)}")
    print(f"  Top 15 signos:")
    for signo, count in freq.most_common(15):
        print(f"    '{signo}'  {count}")

# ─────────────────────────────────────────
# 5. USO DE EMOJIS
# ─────────────────────────────────────────
print("\n" + "="*60)
print("5. USO DE EMOJIS")
print("="*60)

for nombre, df, col_texto in [
    ("SAMUEL GARCIA (posts)"   , df_posts, "text"),
    ("CIUDADANOS (comentarios)", df_com  , "texto")
]:
    todos_emojis = []
    textos_con_emoji = 0
    for texto in df[col_texto].dropna():
        emojis = extraer_emojis(texto)
        todos_emojis.extend(emojis)
        if emojis:
            textos_con_emoji += 1

    freq = Counter(todos_emojis)
    pct  = textos_con_emoji / len(df) * 100 if len(df) > 0 else 0

    print(f"\n--- {nombre} ---")
    print(f"  Textos con emoji     : {textos_con_emoji} ({pct:.1f}%)")
    print(f"  Total emojis usados  : {len(todos_emojis)}")
    print(f"  Emojis unicos        : {len(freq)}")
    if freq:
        print(f"  Top 15 emojis:")
        for emoji, count in freq.most_common(15):
            print(f"    {emoji}  {count}")

# ─────────────────────────────────────────
# 6. GUARDAR RESUMEN EN CSV
# ─────────────────────────────────────────
print("\n" + "="*60)
print("6. GUARDANDO RESUMEN")
print("="*60)

resumen = []
for nombre, df, col_texto in [
    ("Samuel Garcia posts"   , df_posts, "text"),
    ("Ciudadanos comentarios", df_com  , "texto")
]:
    df = df.copy()
    df["num_palabras"]   = df[col_texto].apply(
        lambda x: len(limpiar(str(x)).split()) if isinstance(x, str) else 0
    )
    df["num_emojis"]     = df[col_texto].apply(
        lambda x: len(extraer_emojis(str(x)))
    )
    df["num_puntuacion"] = df[col_texto].apply(
        lambda x: len(extraer_puntuacion(str(x)))
    )

    todos_tokens = []
    for texto in df[col_texto].dropna():
        todos_tokens.extend(tokenizar(texto))

    resumen.append({
        "fuente"               : nombre,
        "total_textos"         : len(df),
        "total_palabras"       : df["num_palabras"].sum(),
        "vocabulario_unico"    : len(set(todos_tokens)),
        "palabras_avg"         : round(df["num_palabras"].mean(), 2),
        "palabras_max"         : df["num_palabras"].max(),
        "palabras_min"         : df["num_palabras"].min(),
        "palabras_std"         : round(df["num_palabras"].std(), 2),
        "emojis_total"         : df["num_emojis"].sum(),
        "emojis_avg"           : round(df["num_emojis"].mean(), 2),
        "puntuacion_total"     : df["num_puntuacion"].sum(),
    })

df_resumen = pd.DataFrame(resumen)
output_dir = os.path.join(base_dir, "data", "processed")
os.makedirs(output_dir, exist_ok=True)
path = os.path.join(output_dir, "estadistico_fb.csv")
df_resumen.to_csv(path, index=False, encoding="utf-8-sig")
print(f"  Guardado en: {path}")
print(df_resumen.to_string(index=False))