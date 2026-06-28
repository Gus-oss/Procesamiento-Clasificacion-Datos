"""
Orquestador principal del pipeline de control de calidad visual.

Ejecuto desde la raíz del proyecto:
    python test/tarea4/run_pipeline.py

Flujo completo:
  1. PySpark extrae features de OpenCV de todas las imágenes de MVTec AD
  2. Entreno y optimizo MLP, CustomCNN y EfficientNetB0 con Optuna
  3. Evalúo los tres modelos y muestro la tabla comparativa
  4. Guardo checkpoints, historial y resultados en data/processed/tarea4/
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Agrego la raíz del proyecto al path para que todos los imports funcionen
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config.tarea4_config as cfg
from processors.tarea4.dataset   import MVTecDataset, make_loaders, build_feature_dataframe, spark_session
from processors.tarea4.models    import build_model
from experiments.tarea4.train    import train_pipeline, get_device
from experiments.tarea4.evaluate import evaluate, compare, save


def main():
    device = get_device()

    # ── 1. Datasets de PyTorch para CNN y EfficientNet ────────────────────────
    print("\n[paso 1] Cargando datasets de PyTorch…")
    ds_train = MVTecDataset(split="train")
    ds_test  = MVTecDataset(split="test")
    class_weights = ds_train.class_weights()

    train_loader, val_loader, test_loader = make_loaders(ds_train, ds_test)

    # ── 2. Features de OpenCV con PySpark (para el MLP) ───────────────────────
    print("\n[paso 2] Extrayendo features con PySpark…")
    spark = spark_session()
    df_train, df_test = build_feature_dataframe(spark)
    spark.stop()

    # Armo tensores a partir del DataFrame de pandas
    X_train = torch.tensor(np.stack(df_train["features"].values), dtype=torch.float32)
    y_train = torch.tensor(df_train["label"].values, dtype=torch.long)
    X_test  = torch.tensor(np.stack(df_test["features"].values),  dtype=torch.float32)
    y_test  = torch.tensor(df_test["label"].values,  dtype=torch.long)
    input_dim = X_train.shape[1]

    # DataLoaders para el MLP (trabaja con vectores, no imágenes)
    from torch.utils.data import TensorDataset, DataLoader
    mlp_cats_train = df_train["category"].tolist()
    mlp_cats_test  = df_test["category"].tolist()

    # Encapsulo categorías junto al tensor para reutilizar la función evaluate
    class _VectorDataset(torch.utils.data.Dataset):
        def __init__(self, X, y, cats):
            self.X, self.y, self.cats = X, y, cats
        def __len__(self): return len(self.y)
        def __getitem__(self, i): return self.X[i], self.y[i], self.cats[i]

    mlp_train_loader = DataLoader(
        _VectorDataset(X_train, y_train, mlp_cats_train),
        batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True,
    )
    mlp_val_split  = int(len(X_train) * cfg.VAL_SPLIT)
    mlp_val_loader = DataLoader(
        _VectorDataset(X_train[-mlp_val_split:], y_train[-mlp_val_split:],
                       mlp_cats_train[-mlp_val_split:]),
        batch_size=cfg.BATCH_SIZE,
    )
    mlp_test_loader = DataLoader(
        _VectorDataset(X_test, y_test, mlp_cats_test),
        batch_size=cfg.BATCH_SIZE,
    )

    # ── 3. Entrenamiento de los tres modelos ──────────────────────────────────
    results = []

    print("\n[paso 3a] MLP sobre features de OpenCV")
    mlp, _ = train_pipeline(
        "mlp", mlp_train_loader, mlp_val_loader, device,
        input_dim=input_dim, class_weights=class_weights,
    )
    results.append(evaluate(mlp, mlp_test_loader, device, "MLP"))

    print("\n[paso 3b] CNN desde cero")
    cnn, _ = train_pipeline(
        "cnn", train_loader, val_loader, device,
        class_weights=class_weights,
    )
    results.append(evaluate(cnn, test_loader, device, "CustomCNN"))

    print("\n[paso 3c] EfficientNet-B0 con fine-tuning")
    effnet, _ = train_pipeline(
        "efficientnet", train_loader, val_loader, device,
        class_weights=class_weights,
    )
    results.append(evaluate(effnet, test_loader, device, "EfficientNetB0"))

    # ── 4. Comparación y guardado ─────────────────────────────────────────────
    print("\n[paso 4] Comparación final")
    compare(results)
    save(results)


if __name__ == "__main__":
    main()
