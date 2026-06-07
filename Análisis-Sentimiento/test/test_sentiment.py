import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from processors.sentiment import SentimentAnalyzer

# ─────────────────────────────────────────
# Cargar comentarios extraídos
# ─────────────────────────────────────────
base_dir      = os.path.dirname(os.path.abspath(__file__))
data_dir      = os.path.join(base_dir, "data", "raw")

# Busca el CSV más reciente de comentarios
csv_files = [f for f in os.listdir(data_dir) if f.startswith("comentarios_sg")]
csv_files.sort(reverse=True)
latest    = os.path.join(data_dir, csv_files[0])

print(f" Cargando: {latest}")
df = pd.read_csv(latest)
print(f"   {len(df)} comentarios cargados")

# ─────────────────────────────────────────
# Correr análisis de sentimientos
# ─────────────────────────────────────────
analyzer = SentimentAnalyzer()
df_result = analyzer.analyze_dataframe(df, text_col="texto")

# ─────────────────────────────────────────
# Guardar resultados
# ─────────────────────────────────────────
analyzer.save(df_result, "sentimiento_sg")

# Vista previa
print("\n Muestra de resultados:")
print(df_result[["texto", "sentimiento_es", "sentimiento_score"]].head(10).to_string())