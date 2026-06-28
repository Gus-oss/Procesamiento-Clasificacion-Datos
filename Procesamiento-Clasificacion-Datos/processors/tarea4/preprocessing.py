"""
Pipeline de extracción de features con OpenCV.
Aplico las mismas técnicas del notebook de clase (Canny, Harris, ORB,
histogramas, LBP) y las concateno en un vector numérico por imagen.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# Permito importar config desde cualquier CWD
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config.tarea4_config as cfg


# ── Carga ──────────────────────────────────────────────────────────────────────

def load_image(path: str, size: tuple = cfg.IMAGE_SIZE) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"No pude leer la imagen: {path}")
    return cv2.resize(img, size)


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def blur(gray: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(gray, cfg.GAUSSIAN_KERNEL, 0)


# ── Features individuales ──────────────────────────────────────────────────────

def histogram_features(gray: np.ndarray) -> np.ndarray:
    """
    Calculo el histograma normalizado y tres estadísticas globales:
    media, desviación estándar y entropía de Shannon.
    La entropía me indica qué tan uniforme es la distribución de intensidades,
    útil para distinguir texturas regulares de superficies dañadas.
    """
    hist = cv2.calcHist([gray], [0], None, [cfg.HIST_BINS], [0, 256]).flatten()
    hist /= hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return np.concatenate([hist, [gray.mean(), gray.std(), entropy]])


def canny_features(blurred: np.ndarray) -> np.ndarray:
    """
    Detecto bordes con Canny (igual que en la sección de monedas del notebook)
    y extraigo densidad, media e intensidad de los píxeles de borde.
    Una pieza defectuosa suele tener bordes irregulares → mayor densidad.
    """
    edges = cv2.Canny(blurred, cfg.CANNY_T1, cfg.CANNY_T2)
    total = edges.size
    return np.array([
        edges.sum() / (255 * total),   # densidad de bordes
        edges.mean(),
        edges.std(),
    ])


def harris_features(gray: np.ndarray) -> np.ndarray:
    """
    Detecto esquinas con Harris (sección de esquinas del notebook).
    En una pieza buena las esquinas son predecibles; un defecto introduce
    esquinas espurias que aumentan el conteo y la respuesta máxima.
    """
    dst = cv2.cornerHarris(np.float32(gray), cfg.HARRIS_BLOCK, 3, cfg.HARRIS_K)
    dst = cv2.dilate(dst, None)
    mask = dst > 0.01 * dst.max()
    return np.array([
        mask.sum() / gray.size,   # densidad de esquinas
        dst.mean(),
        dst.max(),
    ])


def orb_features(gray: np.ndarray) -> np.ndarray:
    """
    Extraigo puntos clave con ORB (sección de puntos clave del notebook).
    ORB es invariante a escala y rotación, por lo que es robusto ante
    pequeñas diferencias de posición entre piezas en la línea de producción.
    """
    orb = cv2.ORB_create(nfeatures=cfg.ORB_KEYPOINTS)
    kps, _ = orb.detectAndCompute(gray, None)
    count = len(kps)
    response = np.mean([k.response for k in kps]) if kps else 0.0
    return np.array([count / cfg.ORB_KEYPOINTS, count / gray.size, response])


def lbp_features(gray: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
    """
    Calculo Local Binary Patterns manualmente para no depender de skimage.
    LBP captura la microestructura de la textura; defectos superficiales
    (rasguños, manchas) alteran el histograma LBP de forma característica.
    """
    lbp = np.zeros_like(gray, dtype=np.uint8)
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        dx, dy = int(round(radius * np.cos(angle))), int(round(radius * np.sin(angle)))
        neighbor = np.roll(np.roll(gray, -dy, axis=0), -dx, axis=1)
        lbp += ((neighbor >= gray).astype(np.uint8)) << i

    hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
    return (hist / hist.sum()).astype(np.float32)


# ── Vector completo (~302 dimensiones) ────────────────────────────────────────

def feature_vector(image_path: str) -> np.ndarray:
    """
    Concateno todas las features en un solo vector float32.
    Composición:
        histograma + estadísticas  → 35
        Canny                      →  3
        Harris                     →  3
        ORB                        →  3
        LBP                        → 256
        ─────────────────────────────
        Total                      → 300
    """
    img  = load_image(image_path)
    gray = to_gray(img)
    return np.concatenate([
        histogram_features(gray),    # 35
        canny_features(blur(gray)),  #  3
        harris_features(gray),       #  3
        orb_features(gray),          #  3
        lbp_features(gray),          # 256
    ]).astype(np.float32)


def preprocess_for_cnn(image_path: str) -> np.ndarray:
    """
    Preparo la imagen para el input de la CNN en formato (C, H, W) float32
    normalizado en [0, 1]. La normalización con estadísticas ImageNet
    la aplico directamente en el Dataset de PyTorch.
    """
    img = load_image(image_path, size=cfg.CNN_INPUT_SIZE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(img_rgb, (2, 0, 1))
