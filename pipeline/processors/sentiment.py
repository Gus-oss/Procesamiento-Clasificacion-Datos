# processors/sentiment.py

import pandas as pd
import os
import sys
from tqdm import tqdm
from transformers import pipeline
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SentimentAnalyzer:

    def __init__(self):
        print("\n Cargando modelo robertuito...")
        self.model = pipeline(
            task       = "text-classification",
            model      = "pysentimiento/robertuito-sentiment-analysis",
            truncation = True,
            max_length = 128
        )
        print("    Modelo cargado")

    # ─────────────────────────────────────────
    # 1. Limpiar texto antes de analizar
    # ─────────────────────────────────────────
    def clean_text(self, text):
        import re
        if not isinstance(text, str):
            return ""
        # Eliminar URLs
        text = re.sub(r"http\S+|www\S+", "", text)
        # Eliminar menciones @usuario
        text = re.sub(r"@\w+", "", text)
        # Eliminar caracteres especiales HTML
        text = re.sub(r"&quot;|&amp;|&lt;|&gt;", "", text)
        # Eliminar emojis problemáticos (opcional)
        text = re.sub(r"[^\w\s\.,!?¿¡áéíóúüñÁÉÍÓÚÜÑ]", " ", text)
        # Eliminar espacios múltiples
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ─────────────────────────────────────────
    # 2. Analizar sentimiento de un texto
    # ─────────────────────────────────────────
    def analyze_one(self, text):
        text_clean = self.clean_text(text)
        if not text_clean:
            return {"label": "NEU", "score": 0.0}
        try:
            result = self.model(text_clean[:512])[0]
            return {
                "label": result["label"],   # POS / NEG / NEU
                "score": round(result["score"], 4)
            }
        except Exception:
            return {"label": "NEU", "score": 0.0}

    # ─────────────────────────────────────────
    # 3. Analizar dataframe completo
    # ─────────────────────────────────────────
    def analyze_dataframe(self, df, text_col="texto"):
        print(f"\n🔍 Analizando sentimiento de {len(df)} comentarios...")

        labels  = []
        scores  = []

        for text in tqdm(df[text_col], desc="Procesando"):
            result = self.analyze_one(text)
            labels.append(result["label"])
            scores.append(result["score"])

        df = df.copy()
        df["sentimiento"]       = labels
        df["sentimiento_score"] = scores

        # Mapeo a español para legibilidad
        mapa = {"POS": "Positivo", "NEG": "Negativo", "NEU": "Neutral"}
        df["sentimiento_es"] = df["sentimiento"].map(mapa)

        print(f"    Análisis completado")
        self._print_summary(df)
        return df

    # ─────────────────────────────────────────
    # 4. Resumen en consola
    # ─────────────────────────────────────────
    def _print_summary(self, df):
        total   = len(df)
        counts  = df["sentimiento_es"].value_counts()
        print(f"\n RESUMEN DE SENTIMIENTOS")
        print(f"{'─'*35}")
        for label, count in counts.items():
            pct = count / total * 100
            bar = "█" * int(pct / 5)
            print(f"  {label:<10} {count:>4} ({pct:>5.1f}%) {bar}")
        print(f"{'─'*35}")
        print(f"  {'Total':<10} {total:>4}")

        # Top 3 comentarios más positivos
        print(f"\n Top 3 comentarios más POSITIVOS:")
        top_pos = df[df["sentimiento"] == "POS"].nlargest(3, "sentimiento_score")
        for _, row in top_pos.iterrows():
            print(f"   ({row['sentimiento_score']}) {row['texto'][:80]}...")

        # Top 3 comentarios más negativos
        print(f"\n Top 3 comentarios más NEGATIVOS:")
        top_neg = df[df["sentimiento"] == "NEG"].nlargest(3, "sentimiento_score")
        for _, row in top_neg.iterrows():
            print(f"   ({row['sentimiento_score']}) {row['texto'][:80]}...")

    # ─────────────────────────────────────────
    # 5. Guardar resultados
    # ─────────────────────────────────────────
    def save(self, df, filename):
        base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "data", "processed")
        os.makedirs(output_dir, exist_ok=True)

        path = os.path.join(output_dir, f"{filename}_{datetime.now().strftime('%Y%m%d')}.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n    Guardado en: {path}")
        return path