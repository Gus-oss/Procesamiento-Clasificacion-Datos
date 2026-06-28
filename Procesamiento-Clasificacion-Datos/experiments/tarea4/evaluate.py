"""
Evaluación comparativa de los tres modelos sobre MVTec AD.

Métricas calculadas:
  - Accuracy, Precision, Recall, F1 (macro y por clase)
  - ROC AUC
  - Especificidad (TN rate) — clave en control de calidad industrial:
      mide cuántas piezas buenas no rechazo innecesariamente.
  - Desglose por categoría de MVTec AD
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config.tarea4_config as cfg


# ── Predicción ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Genero predicciones para todo el DataLoader de una vez.
    Retorno (y_true, y_pred, y_proba_defecto, categorías).
    """
    model.eval()
    y_true, y_pred, y_proba, cats = [], [], [], []

    for imgs, labels, categories in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        probas = torch.softmax(logits, 1)[:, 1]

        y_true.extend(labels.numpy())
        y_pred.extend(logits.argmax(1).cpu().numpy())
        y_proba.extend(probas.cpu().numpy())
        cats.extend(list(categories))

    return np.array(y_true), np.array(y_pred), np.array(y_proba), cats


# ── Métricas ───────────────────────────────────────────────────────────────────

def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict:
    cm  = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = float("nan")

    f1pc  = f1_score(y_true, y_pred, average=None, zero_division=0)
    recpc = recall_score(y_true, y_pred, average=None, zero_division=0)
    prepc = precision_score(y_true, y_pred, average=None, zero_division=0)

    return {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "precision_macro":   float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro":      float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro":          float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc":           float(auc),
        "specificity":       float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1_good":           float(f1pc[0])  if len(f1pc)  > 0 else float("nan"),
        "f1_defect":         float(f1pc[1])  if len(f1pc)  > 1 else float("nan"),
        "recall_defect":     float(recpc[1]) if len(recpc) > 1 else float("nan"),
        "precision_defect":  float(prepc[1]) if len(prepc) > 1 else float("nan"),
        "confusion_matrix":  {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_samples":         int(len(y_true)),
        "n_good":            int((y_true == 0).sum()),
        "n_defect":          int((y_true == 1).sum()),
    }


# ── Evaluación de un modelo ────────────────────────────────────────────────────

def evaluate(
    model:      nn.Module,
    loader:     DataLoader,
    device:     torch.device,
    model_name: str,
) -> Dict:
    """
    Evalúo el modelo y muestro el reporte completo en consola.
    Retorno un dict con métricas globales y desglose por categoría.
    """
    print(f"\n{'─'*50}\nEvaluando: {model_name}\n{'─'*50}")
    model = model.to(device)

    y_true, y_pred, y_proba, cats = predict(model, loader, device)

    global_m = _metrics(y_true, y_pred, y_proba)

    # Desglose por categoría
    cats_arr = np.array(cats)
    per_cat  = {
        cat: _metrics(y_true[cats_arr == cat], y_pred[cats_arr == cat], y_proba[cats_arr == cat])
        for cat in sorted(set(cats))
        if (cats_arr == cat).sum() >= 2
    }

    print(classification_report(y_true, y_pred, target_names=["good", "defect"], digits=4))
    print(f"ROC AUC:       {global_m['roc_auc']:.4f}")
    print(f"Especificidad: {global_m['specificity']:.4f}")

    return {"model": model_name, "global": global_m, "per_category": per_cat}


# ── Tabla comparativa ──────────────────────────────────────────────────────────

def compare(results: List[Dict]) -> None:
    """
    Imprimo una tabla comparativa entre los modelos evaluados.

    En control de calidad industrial priorizo:
      recall_defect  → minimizar piezas defectuosas que pasan como buenas
      especificidad  → minimizar rechazos innecesarios de piezas buenas
    """
    cols = ["Modelo", "Accuracy", "F1 Macro", "F1 Defecto", "Recall Defecto", "Especificidad", "ROC AUC"]
    w = 16
    sep = "─" * (w * len(cols))

    print(f"\n{sep}\nCOMPARACIÓN DE MODELOS — MVTec AD\n{sep}")
    print("".join(c.ljust(w) for c in cols))
    print(sep)

    for r in results:
        g = r["global"]
        print("".join(v.ljust(w) for v in [
            r["model"],
            f"{g['accuracy']:.4f}",
            f"{g['f1_macro']:.4f}",
            f"{g['f1_defect']:.4f}",
            f"{g['recall_defect']:.4f}",
            f"{g['specificity']:.4f}",
            f"{g['roc_auc']:.4f}",
        ]))
    print(sep)

    best = max(results, key=lambda r: r["global"]["f1_defect"])
    print(f"\nMejor modelo (F1 defecto): {best['model']}  →  {best['global']['f1_defect']:.4f}")

    # Desglose por categoría del mejor modelo
    print(f"\nDesglose por categoría — {best['model']}:")
    ch = ["Categoría", "Accuracy", "F1 Defecto", "Recall Defecto", "N"]
    print("".join(c.ljust(w) for c in ch))
    print("─" * (w * len(ch)))
    for cat, m in sorted(best["per_category"].items()):
        print("".join(v.ljust(w) for v in [
            cat,
            f"{m['accuracy']:.4f}",
            f"{m['f1_defect']:.4f}",
            f"{m['recall_defect']:.4f}",
            str(m["n_samples"]),
        ]))


def save(results: List[Dict]) -> str:
    out = Path(cfg.OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    path = str(out / "evaluation_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[info] Resultados guardados en: {path}")
    return path
