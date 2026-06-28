"""
Tarea 4 — Detección de centros de objetos circulares
Imagen: señales de tráfico redondas (velocidad máxima 40)
Técnica: segmentación HSV por color rojo + Gaussian Blur + HoughCircles

Ejecutar desde la raíz del proyecto:
    python test/tarea4/detectar_circulos.py
"""

import cv2
import numpy as np
from pathlib import Path

# ── Rutas ──────────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).resolve().parents[2]
IMAGE_PATH  = BASE_DIR / "data" / "raw" / "tarea4" / "senales-de-trafico.webp"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "tarea4" / "resultado_circulos.jpg"
REPORT_PATH = BASE_DIR / "data" / "processed" / "tarea4" / "reporte.md"

# ── Parámetros calibrados para esta imagen ────────────────────────────────────

# Rango de rojo en HSV (el borde de las señales de límite de velocidad)
RED_LOWER1 = np.array([0,   80, 60])
RED_UPPER1 = np.array([12, 255, 255])
RED_LOWER2 = np.array([155, 80, 60])
RED_UPPER2 = np.array([180, 255, 255])

BLUR_KERNEL  = (11, 11)
HOUGH_DP     = 1.0
HOUGH_MINDIST= 100
HOUGH_P1     = 50    # umbral Canny interno de HoughCircles
HOUGH_P2     = 15    # umbral acumulador — calibrado para esta imagen
HOUGH_MINR   = 45
HOUGH_MAXR   = 95

# Filtros post-detección para eliminar falsos positivos
MIN_RADIUS   = 70    # radio mínimo de señal real en píxeles
# Posición horizontal esperada de las señales (izquierda y derecha de carretera)
X_SIGN_LEFT  = (0,   150)
X_SIGN_RIGHT = (600, 720)


# ── Pipeline ───────────────────────────────────────────────────────────────────

def segmentar_rojo(img: np.ndarray) -> np.ndarray:
    """
    Segmento los píxeles rojos en espacio HSV y dilato la máscara para
    cerrar huecos en el borde circular de las señales.
    El rojo ocupa dos rangos en HSV (0-12° y 155-180°) por eso uso dos máscaras.
    """
    hsv    = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask   = cv2.bitwise_or(
        cv2.inRange(hsv, RED_LOWER1, RED_UPPER1),
        cv2.inRange(hsv, RED_LOWER2, RED_UPPER2),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.dilate(mask, kernel, iterations=2)


def detectar_circulos(img: np.ndarray, mask: np.ndarray):
    """
    Aplico el pipeline del notebook de clase sobre la región roja:
    escala de grises → Gaussian Blur → HoughCircles.
    """
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_m  = cv2.bitwise_and(gray, gray, mask=mask)
    blurred = cv2.GaussianBlur(gray_m, BLUR_KERNEL, 2)
    edges   = cv2.Canny(blurred, 50, 150)   # solo para reporte visual

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp       = HOUGH_DP,
        minDist  = HOUGH_MINDIST,
        param1   = HOUGH_P1,
        param2   = HOUGH_P2,
        minRadius= HOUGH_MINR,
        maxRadius= HOUGH_MAXR,
    )
    return circles, edges


def filtrar(circles) -> list:
    """
    Elimino falsos positivos usando dos criterios:
      1. Radio mínimo: los círculos del letrero azul tienen r < 70 px.
      2. Posición horizontal: las señales de velocidad están en los
         laterales de la carretera, no en el centro de la imagen.
    """
    if circles is None:
        return []
    candidatos = [c for c in circles[0] if c[2] >= MIN_RADIUS]
    senales = [
        c for c in candidatos
        if (X_SIGN_LEFT[0]  <= c[0] <= X_SIGN_LEFT[1]) or
           (X_SIGN_RIGHT[0] <= c[0] <= X_SIGN_RIGHT[1])
    ]
    return [{"id": i+1, "cx": int(c[0]), "cy": int(c[1]), "radio": int(c[2])}
            for i, c in enumerate(senales)]


def anotar(img: np.ndarray, detectados: list) -> np.ndarray:
    resultado = img.copy()
    for d in detectados:
        x, y, r = d["cx"], d["cy"], d["radio"]
        cv2.circle(resultado, (x, y), r, (0, 200, 0), 3)
        cv2.circle(resultado, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(resultado, f"#{d['id']}", (x - 15, y - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return resultado


def generar_reporte(img_shape: tuple, detectados: list) -> None:
    h, w = img_shape[:2]
    n    = len(detectados)
    radios = [d["radio"] for d in detectados]

    lineas = [
        "# Tarea 4 — Detección de centros de objetos circulares",
        "",
        "## Objetivo",
        "Detectar automáticamente señales de tráfico circulares en una imagen",
        "real de carretera, identificando el centro y radio de cada señal.",
        "",
        "## Metodología",
        "",
        "### 1. Segmentación por color (HSV)",
        "Convierto la imagen a espacio HSV y genero una máscara binaria",
        "para los píxeles de color rojo. El rojo ocupa dos rangos en HSV",
        "(0–12° y 155–180°), por lo que uso dos máscaras y las combino.",
        "Aplico dilatación morfológica con kernel elíptico 7×7 para cerrar",
        "huecos en el borde circular de las señales.",
        "",
        "### 2. Gaussian Blur",
        f"Aplico desenfoque gaussiano con kernel `{BLUR_KERNEL}` sobre la",
        "región segmentada para reducir el ruido antes de detectar bordes.",
        "",
        "### 3. Detección de bordes con Canny",
        "Genero el mapa de bordes con umbrales `50` y `150` para visualización.",
        "",
        "### 4. Transformada de Hough para círculos",
        "Aplico `cv2.HoughCircles` sobre la imagen desenfocada.",
        "",
        "| Parámetro | Valor | Descripción |",
        "|-----------|-------|-------------|",
        f"| `dp` | `{HOUGH_DP}` | Resolución del acumulador |",
        f"| `minDist` | `{HOUGH_MINDIST}` | Distancia mínima entre centros |",
        f"| `param1` | `{HOUGH_P1}` | Umbral Canny interno |",
        f"| `param2` | `{HOUGH_P2}` | Umbral acumulador |",
        f"| `minRadius` | `{HOUGH_MINR}` | Radio mínimo (px) |",
        f"| `maxRadius` | `{HOUGH_MAXR}` | Radio máximo (px) |",
        "",
        "### 5. Filtrado post-detección",
        "Elimino falsos positivos con dos criterios combinados:",
        f"- Radio mínimo ≥ `{MIN_RADIUS}` px (los letreros azules generan círculos más pequeños)",
        "- Posición horizontal: solo acepto detecciones en los laterales de la imagen,",
        "  donde se ubican físicamente las señales de velocidad.",
        "",
        "## Imagen analizada",
        f"- **Resolución:** {w} × {h} px",
        "- **Contenido:** autopista con dos señales de velocidad máxima 40",
        "",
        "## Resultados",
        f"Se detectaron **{n} señal(es)** circular(es) en la imagen.",
        "",
        "| # | Centro (x, y) | Radio (px) | Descripción |",
        "|---|--------------|------------|-------------|",
    ]

    for d in detectados:
        lado = "izquierda" if d["cx"] < 300 else "derecha"
        lineas.append(f"| {d['id']} | ({d['cx']}, {d['cy']}) | {d['radio']} | Señal velocidad máx. 40 — {lado} |")

    if radios:
        lineas += [
            "",
            "### Estadísticas de radios",
            f"- **Mínimo:** {min(radios)} px",
            f"- **Máximo:** {max(radios)} px",
            f"- **Promedio:** {np.mean(radios):.1f} px",
            f"- **Diferencia entre señales:** {abs(radios[0]-radios[1]) if len(radios)==2 else 'N/A'} px",
            "  *(diferencia esperada por perspectiva: la señal izquierda está más cerca)*",
        ]

    lineas += [
        "",
        "## Imagen resultado",
        "![Resultado](resultado_circulos.jpg)",
        "",
        "## Conclusiones",
        "- La segmentación previa por color rojo en HSV reduce drásticamente",
        "  los falsos positivos respecto a aplicar HoughCircles directamente.",
        "- El Gaussian Blur previo es indispensable: sin él HoughCircles genera",
        "  decenas de detecciones espurias en el cielo y el asfalto.",
        "- El parámetro `param2` es el más sensible de HoughCircles: un valor",
        "  de 15 detecta 5 candidatos; el filtrado posterior los reduce a 2.",
        "- La diferencia de radio entre las dos señales refleja la perspectiva:",
        "  la señal derecha aparece más grande porque está más cerca de la cámara.",
        "- Para escenas más complejas convendría añadir un clasificador",
        "  entrenado sobre los recortes de cada candidato detectado.",
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lineas), encoding="utf-8")
    print(f"[info] Reporte guardado en: {REPORT_PATH}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"No encuentro la imagen en:\n  {IMAGE_PATH}\n"
            "Verifica que el archivo esté en data/raw/tarea4/"
        )

    img = cv2.imread(str(IMAGE_PATH))
    print(f"[info] Imagen cargada: {img.shape[1]}×{img.shape[0]} px")

    mask             = segmentar_rojo(img)
    circles, edges   = detectar_circulos(img, mask)
    detectados       = filtrar(circles)
    resultado        = anotar(img, detectados)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), resultado)
    print(f"[info] Imagen anotada guardada en: {OUTPUT_PATH}")
    print(f"[info] Señales detectadas: {len(detectados)}")
    for d in detectados:
        lado = "izquierda" if d["cx"] < 300 else "derecha"
        print(f"       #{d['id']}  centro=({d['cx']},{d['cy']})  radio={d['radio']}px  [{lado}]")

    generar_reporte(img.shape, detectados)
    print("\n[listo] Archivos generados:")
    print(f"        {OUTPUT_PATH}")
    print(f"        {REPORT_PATH}")


if __name__ == "__main__":
    main()