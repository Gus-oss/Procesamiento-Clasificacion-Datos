"""
Carga distribuida de MVTec AD con PySpark y Dataset de PyTorch para las CNNs.

Flujo:
  1. PySpark escanea todas las categorías en paralelo y extrae el
     vector de features de OpenCV por imagen → DataFrame de pandas.
  2. MVTecDataset carga imágenes directamente desde disco con augmentation
     para el entrenamiento de las redes convolucionales.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType, IntegerType, StringType, StructField, StructType

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config.tarea4_config as cfg
from processors.tarea4.preprocessing import feature_vector


# ── PySpark: carga distribuida ─────────────────────────────────────────────────

def spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName(cfg.SPARK_APP_NAME)
        .master("local[*]")
        .config("spark.driver.memory", cfg.SPARK_DRIVER_MEM)
        .config("spark.sql.shuffle.partitions", cfg.SPARK_PARTITIONS)
        .getOrCreate()
    )


def _discover(root: Path, categories: List[str]) -> List[dict]:
    """
    Escaneo el disco y construyo la lista de registros.
    MVTec AD usa esta convención:
        train/good/          → label 0
        test/good/           → label 0
        test/<defecto>/      → label 1
    """
    records = []
    for cat in categories:
        cat_dir = root / cat
        if not cat_dir.exists():
            print(f"[aviso] No encuentro la categoría en disco: {cat_dir}")
            continue
        for split in ("train", "test"):
            split_dir = cat_dir / split
            if not split_dir.exists():
                continue
            for defect_dir in split_dir.iterdir():
                if not defect_dir.is_dir():
                    continue
                label = 0 if defect_dir.name == "good" else 1
                for p in defect_dir.glob("*.png"):
                    records.append({
                        "path": str(p),
                        "category": cat,
                        "split": split,
                        "defect": defect_dir.name,
                        "label": label,
                    })
    print(f"[info] Imágenes encontradas: {len(records)}")
    return records


def build_feature_dataframe(spark: SparkSession) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extraigo el vector de features para cada imagen en paralelo con PySpark
    y separo el resultado en train y test como DataFrames de pandas.
    """
    records = _discover(cfg.MVTEC_ROOT, cfg.CATEGORIES)

    schema = StructType([
        StructField("path",     StringType(),  False),
        StructField("category", StringType(),  False),
        StructField("split",    StringType(),  False),
        StructField("defect",   StringType(),  False),
        StructField("label",    IntegerType(), False),
    ])
    df = spark.createDataFrame(
        [(r["path"], r["category"], r["split"], r["defect"], r["label"]) for r in records],
        schema,
    ).repartition(cfg.SPARK_PARTITIONS)

    @F.udf(returnType=ArrayType(FloatType()))
    def _extract(path: str) -> Optional[List[float]]:
        # Importo dentro de la UDF porque cada worker de Spark corre
        # en su propio proceso y no hereda el namespace del driver.
        try:
            import sys, pathlib
            sys.path.insert(0, str(pathlib.Path(path).parents[4]))
            from processors.tarea4.preprocessing import feature_vector
            return feature_vector(path).tolist()
        except Exception as e:
            print(f"[error UDF] {path}: {e}")
            return None

    print("[info] Extrayendo features con PySpark…")
    df = (
        df.withColumn("features", _extract(F.col("path")))
          .filter(F.col("features").isNotNull())
    )

    pdf = df.toPandas()
    pdf["features"] = pdf["features"].apply(np.array)
    print(f"[info] Features listas: {len(pdf)} imágenes.")

    return (
        pdf[pdf["split"] == "train"].reset_index(drop=True),
        pdf[pdf["split"] == "test"].reset_index(drop=True),
    )


# ── PyTorch Dataset ────────────────────────────────────────────────────────────

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

TRAIN_TF = T.Compose([
    T.ToPILImage(),
    T.Resize(cfg.CNN_INPUT_SIZE),
    T.RandomHorizontalFlip(0.5),
    T.RandomVerticalFlip(0.3),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    T.ToTensor(),
    T.Normalize(_MEAN, _STD),
])

VAL_TF = T.Compose([
    T.ToPILImage(),
    T.Resize(cfg.CNN_INPUT_SIZE),
    T.ToTensor(),
    T.Normalize(_MEAN, _STD),
])


class MVTecDataset(Dataset):
    """
    Dataset de PyTorch para MVTec AD.
    Cada muestra retorna (tensor_imagen, label_binario, nombre_categoria).
    """

    def __init__(self, split: str = "train", transform=None):
        self.transform = transform or (TRAIN_TF if split == "train" else VAL_TF)
        self.samples   = self._scan(split)
        print(f"[info] MVTecDataset ({split}): {len(self.samples)} muestras")

    def _scan(self, split: str) -> List[Tuple[str, int, str]]:
        samples = []
        for cat in cfg.CATEGORIES:
            split_dir = cfg.MVTEC_ROOT / cat / split
            if not split_dir.exists():
                continue
            for defect_dir in split_dir.iterdir():
                if not defect_dir.is_dir():
                    continue
                label = 0 if defect_dir.name == "good" else 1
                for p in defect_dir.glob("*.png"):
                    samples.append((str(p), label, cat))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path, label, cat = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"No pude leer: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img), label, cat

    def class_weights(self) -> torch.Tensor:
        """
        Calculo pesos inversos de clase para compensar el desbalance natural
        de MVTec AD (muchas más muestras buenas que defectuosas en train).
        """
        labels = [s[1] for s in self.samples]
        n = len(labels)
        n0, n1 = labels.count(0), labels.count(1)
        w0 = n / (2 * n0) if n0 else 1.0
        w1 = n / (2 * n1) if n1 else 1.0
        return torch.tensor([w0, w1], dtype=torch.float32)


def make_loaders(
    ds_train: MVTecDataset,
    ds_test:  MVTecDataset,
    batch_size: int = cfg.BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Separo una fracción de entrenamiento para validación y devuelvo
    los tres DataLoaders listos para usar.
    """
    n_val   = int(len(ds_train) * cfg.VAL_SPLIT)
    n_train = len(ds_train) - n_val
    gen     = torch.Generator().manual_seed(cfg.SEED)
    train_sub, val_sub = random_split(ds_train, [n_train, n_val], generator=gen)

    kwargs = dict(batch_size=batch_size, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    return (
        DataLoader(train_sub, shuffle=True,  drop_last=True, **kwargs),
        DataLoader(val_sub,   shuffle=False, **kwargs),
        DataLoader(ds_test,   shuffle=False, **kwargs),
    )
