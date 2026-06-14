# experiments/experiment.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection   import StratifiedKFold, cross_validate
from sklearn.pipeline          import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm               import SVC
from sklearn.naive_bayes       import MultinomialNB
from sklearn.preprocessing     import LabelEncoder
from sklearn.metrics           import (accuracy_score, precision_score,
                                       recall_score, f1_score,
                                       classification_report,
                                       confusion_matrix)
from transformers              import pipeline as hf_pipeline
import torch

# ─────────────────────────────────────────
# CLASE PRINCIPAL DEL EXPERIMENTO
# ─────────────────────────────────────────
class TextClassificationExperiment:

    def __init__(self, data_dir, output_dir):
        self.data_dir   = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ─────────────────────────────────────────
    # 1. CARGA Y COMBINACION DE DATOS
    # ─────────────────────────────────────────
    def load_data(self):
        print("\n" + "="*60)
        print("CARGANDO DATOS")
        print("="*60)

        # Facebook
        fb_file = sorted([
            f for f in os.listdir(self.data_dir)
            if f.startswith("sentimiento_samuel_fb")
        ])[-1]
        df_fb = pd.read_csv(os.path.join(self.data_dir, fb_file))
        df_fb["fuente"] = "facebook"
        print(f"Facebook : {len(df_fb)} comentarios ({fb_file})")

        # YouTube
        yt_file = sorted([
            f for f in os.listdir(self.data_dir)
            if f.startswith("sentimiento_sg_yt")
        ])[-1]
        df_yt = pd.read_csv(os.path.join(self.data_dir, yt_file))
        df_yt["fuente"] = "youtube"
        print(f"YouTube  : {len(df_yt)} comentarios ({yt_file})")

        # Combinar
        df = pd.concat(
            [df_fb[["texto", "sentimiento", "sentimiento_es", "fuente"]],
             df_yt[["texto", "sentimiento", "sentimiento_es", "fuente"]]],
            ignore_index=True
        )

        # Limpiar nulos y textos vacios
        df = df[df["texto"].notna()]
        df = df[df["texto"].str.strip() != ""]
        df = df[df["sentimiento"].isin(["POS", "NEG", "NEU"])]
        df = df.reset_index(drop=True)

        print(f"\nDataset combinado: {len(df)} comentarios")
        print(f"Distribucion de clases:")
        counts = df["sentimiento"].value_counts()
        for label, count in counts.items():
            pct = count / len(df) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")

        self.df = df
        self.X  = df["texto"].tolist()
        self.y  = df["sentimiento"].tolist()
        return df

    # ─────────────────────────────────────────
    # 2. EVALUAR MODELO CLASICO (SVM / NB)
    # ─────────────────────────────────────────
    def evaluate_classic(self, model_name, pipeline, cv_folds=5):
        print(f"\nEvaluando: {model_name}")

        le    = LabelEncoder()
        y_enc = le.fit_transform(self.y)

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        scoring = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]

        t_inicio = time.time()
        scores   = cross_validate(
            pipeline, self.X, y_enc,
            cv=cv, scoring=scoring, n_jobs=-1
        )
        t_total  = time.time() - t_inicio

        resultado = {
            "modelo"          : model_name,
            "tipo"            : "clasico",
            "accuracy_avg"    : round(scores["test_accuracy"].mean(), 4),
            "accuracy_std"    : round(scores["test_accuracy"].std(), 4),
            "f1_macro_avg"    : round(scores["test_f1_macro"].mean(), 4),
            "f1_macro_std"    : round(scores["test_f1_macro"].std(), 4),
            "precision_avg"   : round(scores["test_precision_macro"].mean(), 4),
            "recall_avg"      : round(scores["test_recall_macro"].mean(), 4),
            "tiempo_seg"      : round(t_total, 2),
            "folds"           : cv_folds
        }

        print(f"  Accuracy : {resultado['accuracy_avg']:.4f} "
              f"(+/- {resultado['accuracy_std']:.4f})")
        print(f"  F1-macro : {resultado['f1_macro_avg']:.4f} "
              f"(+/- {resultado['f1_macro_std']:.4f})")
        print(f"  Tiempo   : {resultado['tiempo_seg']}s")

        return resultado

    # ─────────────────────────────────────────
    # 3. EVALUAR MODELO TRANSFORMER
    # ─────────────────────────────────────────
    def evaluate_transformer(self, model_name, model_id,
                              label_map, cv_folds=5):
        print(f"\nEvaluando: {model_name}")
        print(f"  Cargando modelo {model_id}...")

        device = 0 if torch.cuda.is_available() else -1

        clf = hf_pipeline(
            task      = "text-classification",
            model     = model_id,
            truncation= True,
            max_length= 128,
            device    = device
        )

        cv        = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        X_arr     = np.array(self.X)
        y_arr     = np.array(self.y)

        fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(cv.split(X_arr, y_arr)):
            print(f"  Fold {fold+1}/{cv_folds}...", end=" ", flush=True)

            X_test = X_arr[test_idx].tolist()
            y_test = y_arr[test_idx].tolist()

            t_inicio = time.time()
            preds_raw = clf(X_test, batch_size=16)
            t_fold    = time.time() - t_inicio

            y_pred = [label_map.get(p["label"], "NEU") for p in preds_raw]

            acc  = accuracy_score(y_test, y_pred)
            f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)

            fold_metrics.append({
                "accuracy" : acc,
                "f1_macro" : f1,
                "precision": prec,
                "recall"   : rec,
                "tiempo"   : t_fold
            })
            print(f"acc={acc:.3f} f1={f1:.3f} ({t_fold:.1f}s)")

        df_folds = pd.DataFrame(fold_metrics)

        resultado = {
            "modelo"        : model_name,
            "tipo"          : "transformer",
            "accuracy_avg"  : round(df_folds["accuracy"].mean(), 4),
            "accuracy_std"  : round(df_folds["accuracy"].std(), 4),
            "f1_macro_avg"  : round(df_folds["f1_macro"].mean(), 4),
            "f1_macro_std"  : round(df_folds["f1_macro"].std(), 4),
            "precision_avg" : round(df_folds["precision"].mean(), 4),
            "recall_avg"    : round(df_folds["recall"].mean(), 4),
            "tiempo_seg"    : round(df_folds["tiempo"].sum(), 2),
            "folds"         : cv_folds
        }

        print(f"  Accuracy : {resultado['accuracy_avg']:.4f} "
              f"(+/- {resultado['accuracy_std']:.4f})")
        print(f"  F1-macro : {resultado['f1_macro_avg']:.4f} "
              f"(+/- {resultado['f1_macro_std']:.4f})")
        print(f"  Tiempo   : {resultado['tiempo_seg']}s")

        return resultado

    # ─────────────────────────────────────────
    # 4. REPORTE FINAL POR MODELO (ultima iteracion)
    # ─────────────────────────────────────────
    def full_report(self, model_name, y_true, y_pred):
        print(f"\nReporte completo: {model_name}")
        print(classification_report(y_true, y_pred,
              target_names=["NEG", "NEU", "POS"]))
        cm = confusion_matrix(y_true, y_pred, labels=["NEG", "NEU", "POS"])
        df_cm = pd.DataFrame(
            cm,
            index  =["Real NEG", "Real NEU", "Real POS"],
            columns=["Pred NEG", "Pred NEU", "Pred POS"]
        )
        return df_cm

    # ─────────────────────────────────────────
    # 5. GUARDAR RESULTADOS EN EXCEL
    # ─────────────────────────────────────────
    def save_results(self, resultados, confusion_matrices):
        output_path = os.path.join(self.output_dir, "experimento_clasificacion.xlsx")

        df_res = pd.DataFrame(resultados).sort_values(
            "f1_macro_avg", ascending=False
        )

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:

            # Hoja 1 — Comparativa general
            df_res.to_excel(writer, sheet_name="1_Comparativa", index=False)

            # Hoja 2 — Ranking por F1
            df_rank = df_res[["modelo", "f1_macro_avg", "f1_macro_std",
                               "accuracy_avg", "precision_avg",
                               "recall_avg", "tiempo_seg"]].copy()
            df_rank.insert(0, "ranking", range(1, len(df_rank)+1))
            df_rank.to_excel(writer, sheet_name="2_Ranking", index=False)

            # Hojas 3+ — Matrices de confusion por modelo
            for modelo, df_cm in confusion_matrices.items():
                nombre_hoja = f"CM_{modelo[:25]}"
                df_cm.to_excel(writer, sheet_name=nombre_hoja)

        print(f"\nResultados guardados en: {output_path}")
        return output_path