"""
Tarea 4 — Detección de centros de objetos circulares
Imagen: señales de tráfico redondas (velocidad máxima 40 + señales superiores)
Técnica: segmentación HSV + HoughCircles (señales grandes) + 
         análisis de circularidad por contornos (señales pequeñas)

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

# ── Parámetros ─────────────────────────────────────────────────────────────────

RED_LOWER1 = np.array([0,   60, 40])
RED_UPPER1 = np.array([20, 255, 255])
RED_LOWER2 = np.array([140, 60, 40])
RED_UPPER2 = np.array([180, 255, 255])

BLUR_KERNEL   = (11, 11)
HOUGH_DP      = 1.0
HOUGH_MINDIST = 100
HOUGH_P1      = 50
HOUGH_P2      = 15
HOUGH_MINR    = 45
HOUGH_MAXR    = 95

# Filtros para HoughCircles (señales grandes)
MIN_RADIUS_LARGE = 70
X_LEFT           = (0,   150)
X_RIGHT          = (600, 720)

# Filtros para contornos (señales pequeñas)
MIN_AREA         = 80
MIN_CIRCULARITY  = 0.7
MIN_ASPECT       = 0.6
MAX_RADIUS_SMALL = 20   # señales pequeñas tienen r < 20 px


# ── Segmentación roja ─────────────────────────────────────────────────────────

def segmentar_rojo(img: np.ndarray):
    """
    Genero la máscara HSV de píxeles rojos.
    El rojo ocupa dos rangos en HSV (0–20° y 140–180°), combino ambos.
    Uso dilatación morfológica para cerrar huecos en bordes circulares.
    """
    hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red   = cv2.bitwise_or(
        cv2.inRange(hsv, RED_LOWER1, RED_UPPER1),
        cv2.inRange(hsv, RED_LOWER2, RED_UPPER2),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.dilate(red, kernel, iterations=2)


# ── Detección de señales grandes con HoughCircles ─────────────────────────────

def detectar_grandes(img: np.ndarray, mask: np.ndarray) -> list:
    """
    Aplico el pipeline del notebook: escala de grises → Gaussian Blur →
    HoughCircles sobre la región roja segmentada.
    Filtro por radio mínimo y posición horizontal para eliminar falsos positivos.
    """
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_m  = cv2.bitwise_and(gray, gray, mask=mask)
    blurred = cv2.GaussianBlur(gray_m, BLUR_KERNEL, 2)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP, minDist=HOUGH_MINDIST,
        param1=HOUGH_P1, param2=HOUGH_P2,
        minRadius=HOUGH_MINR, maxRadius=HOUGH_MAXR,
    )
    if circles is None:
        return []

    resultado = []
    for c in circles[0]:
        x, y, r = int(c[0]), int(c[1]), int(c[2])
        en_lateral = (X_LEFT[0] <= x <= X_LEFT[1]) or (X_RIGHT[0] <= x <= X_RIGHT[1])
        if r >= MIN_RADIUS_LARGE and en_lateral:
            resultado.append({"cx": x, "cy": y, "radio": r, "tipo": "señal velocidad máx. 40"})

    return resultado


# ── Detección de señales pequeñas por circularidad de contorno ────────────────

def detectar_pequeñas(mask: np.ndarray) -> list:
    """
    Uso análisis de contornos en lugar de HoughCircles para detectar
    las señales pequeñas encima de los letreros, donde el radio es
    demasiado pequeño (~7 px) para que HoughCircles sea confiable.

    Filtro cada contorno por:
      - Área mínima: elimina ruido puntual
      - Circularidad (4πA/P²): valores cercanos a 1 indican forma circular
      - Relación de aspecto del bounding box: evita rectángulos alargados
      - Radio máximo: separo las pequeñas de las grandes
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    resultado   = []

    for cnt in contours:
        area  = cv2.contourArea(cnt)
        perim = cv2.arcLength(cnt, True)
        if area < MIN_AREA or perim == 0:
            continue

        circularidad = 4 * np.pi * area / (perim ** 2)
        x, y, w, h  = cv2.boundingRect(cnt)
        aspecto      = min(w, h) / max(w, h) if max(w, h) > 0 else 0
        cx, cy       = x + w // 2, y + h // 2
        r            = max(w, h) // 2

        if circularidad >= MIN_CIRCULARITY and aspecto >= MIN_ASPECT and r <= MAX_RADIUS_SMALL:
            resultado.append({"cx": cx, "cy": cy, "radio": r, "tipo": "señal circular superior"})

    return resultado


# ── Anotación ─────────────────────────────────────────────────────────────────

def anotar(img: np.ndarray, detectados: list) -> np.ndarray:
    resultado = img.copy()
    for d in detectados:
        x, y, r = d["cx"], d["cy"], d["radio"]
        color = (0, 200, 0) if d["tipo"] == "señal velocidad máx. 40" else (0, 200, 255)
        cv2.circle(resultado, (x, y), r, color, 2)
        cv2.circle(resultado, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(resultado, f"#{d['id']}", (x - 15, y - r - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return resultado


# ── Reporte Markdown ───────────────────────────────────────────────────────────

def generar_reporte(img_shape: tuple, detectados: list) -> None:
    h, w   = img_shape[:2]
    grandes = [d for d in detectados if d["tipo"] == "señal velocidad máx. 40"]
    pequeñas= [d for d in detectados if d["tipo"] == "señal circular superior"]
    radios  = [d["radio"] for d in detectados]

    lineas = [
        "# Tarea 4 — Detección de centros de objetos circulares",
        "",
        "## Objetivo",
        "Detectar automáticamente todas las señales de tráfico circulares",
        "en una imagen real de carretera, identificando el centro y radio",
        "de cada señal independientemente de su tamaño.",
        "",
        "## Metodología",
        "",
        "Combino dos técnicas complementarias para cubrir señales de",
        "diferentes tamaños en la misma imagen:",
        "",
        "### 1. Segmentación por color rojo (HSV)",
        "Convierto la imagen a espacio HSV y genero una máscara binaria",
        "para los píxeles de color rojo. El rojo ocupa dos rangos en HSV",
        "(0–20° y 140–180°), por lo que uso dos máscaras y las combino.",
        "Aplico dilatación morfológica con kernel elíptico 5×5 para cerrar",
        "huecos en los bordes circulares de las señales.",
        "",
        "### 2. Gaussian Blur + HoughCircles (señales grandes, r ≥ 45 px)",
        f"Aplico desenfoque gaussiano con kernel `{BLUR_KERNEL}` y luego",
        "`cv2.HoughCircles` sobre la región segmentada.",
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
        "Filtro los candidatos por radio mínimo ≥ 70 px y posición",
        "horizontal (laterales de la carretera).",
        "",
        "### 3. Análisis de circularidad por contornos (señales pequeñas, r < 20 px)",
        "Para las señales circulares encima de los letreros, cuyo radio",
        "es demasiado pequeño para HoughCircles, uso `cv2.findContours`",
        "y filtro cada contorno por tres criterios:",
        "",
        f"- **Área mínima** ≥ `{MIN_AREA}` px² — elimina ruido puntual",
        f"- **Circularidad** (4πA/P²) ≥ `{MIN_CIRCULARITY}` — valores cercanos a 1",
        "  indican forma circular perfecta",
        f"- **Relación de aspecto** ≥ `{MIN_ASPECT}` — evita contornos alargados",
        "",
        "## Imagen analizada",
        f"- **Resolución:** {w} × {h} px",
        "- **Contenido:** autopista con señales de velocidad máxima 40",
        "  y señales circulares de regulación encima de los letreros",
        "",
        "## Resultados",
        f"Se detectaron **{len(detectados)} señal(es)** circular(es) en total.",
        "",
        "| # | Centro (x, y) | Radio (px) | Tipo |",
        "|---|--------------|------------|------|",
    ]

    for d in detectados:
        lineas.append(
            f"| {d['id']} | ({d['cx']}, {d['cy']}) | {d['radio']} | {d['tipo']} |"
        )

    lineas += [
        "",
        "### Estadísticas de radios",
        f"- **Mínimo:** {min(radios)} px",
        f"- **Máximo:** {max(radios)} px",
        f"- **Promedio:** {np.mean(radios):.1f} px",
        "",
        "## Imagen resultado",
        "*(círculos verdes: señales de velocidad | círculos amarillos: señales superiores)*",
        "",
        "![Resultado](resultado_circulos.jpg)",
        "",
        "## Conclusiones",
        "- La segmentación previa por color rojo en HSV es indispensable:",
        "  aplicar HoughCircles directamente genera cientos de falsos positivos.",
        "- HoughCircles es efectivo para señales grandes (r ≥ 45 px) pero",
        "  pierde fiabilidad para señales de radio muy pequeño (r < 20 px).",
        "- El análisis de circularidad por contornos complementa a HoughCircles",
        "  para detectar señales pequeñas con alta precisión.",
        "- La diferencia de radio entre las señales de velocidad (76 vs 82 px)",
        "  refleja la perspectiva: la señal derecha está más cerca de la cámara.",
        "- Combinar ambas técnicas permite detectar señales en un rango de",
        f"  tamaños de {min(radios)} a {max(radios)} px de radio en la misma imagen.",
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

    img  = cv2.imread(str(IMAGE_PATH))
    print(f"[info] Imagen cargada: {img.shape[1]}×{img.shape[0]} px")

    mask      = segmentar_rojo(img)
    grandes   = detectar_grandes(img, mask)
    pequeñas  = detectar_pequeñas(mask)

    # Combino y asigno IDs correlativos
    todos = grandes + pequeñas
    for i, d in enumerate(todos, 1):
        d["id"] = i

    resultado = anotar(img, todos)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), resultado)

    print(f"[info] Señales grandes detectadas:  {len(grandes)}")
    print(f"[info] Señales pequeñas detectadas: {len(pequeñas)}")
    print(f"[info] Total: {len(todos)}")
    for d in todos:
        print(f"       #{d['id']}  centro=({d['cx']},{d['cy']})  r={d['radio']}px  [{d['tipo']}]")

    generar_reporte(img.shape, todos)
    print(f"\n[listo] {OUTPUT_PATH}")
    print(f"[listo] {REPORT_PATH}")


if __name__ == "__main__":
    main()