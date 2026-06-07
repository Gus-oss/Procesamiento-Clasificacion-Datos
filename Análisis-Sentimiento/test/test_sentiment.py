# test/test_sentiment.py
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from processors.sentiment import SentimentAnalyzer

# ─────────────────────────────────────────
# Rutas correctas
# ─────────────────────────────────────────
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data", "raw")

print(f"base_dir : {base_dir}")   # para verificar que apunta bien
print(f"data_dir : {data_dir}")

# Busca el CSV mas reciente de comentarios de YouTube
csv_files = [f for f in os.listdir(data_dir) if f.startswith("comentarios_sg_20260607")]
csv_files.sort(reverse=True)

if not csv_files:
    print("No se encontraron archivos de comentarios en data/raw/")
    print(f"Archivos disponibles: {os.listdir(data_dir)}")
    sys.exit(1)

latest = os.path.join(data_dir, csv_files[0])
print(f"Cargando: {latest}")

df = pd.read_csv(latest)
print(f"   {len(df)} comentarios cargados")

# Filtrar comentarios vacios
df = df[df["texto"].notna() & (df["texto"].str.strip() != "")]
print(f"   {len(df)} comentarios validos para analizar")

# ─────────────────────────────────────────
# Correr analisis de sentimientos
# ─────────────────────────────────────────
analyzer  = SentimentAnalyzer()
df_result = analyzer.analyze_dataframe(df, text_col="texto")

# ─────────────────────────────────────────
# Guardar resultados
# ─────────────────────────────────────────
analyzer.save(df_result, "sentimiento_adg")

# Vista previa
print("\nMuestra de resultados:")
print(df_result[["texto", "sentimiento_es", "sentimiento_score"]].head(10).to_string())