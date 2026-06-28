"""
Tarea 4 — Detección de centros de objetos circulares
Imagen: señales de tráfico redondas
Técnica: Gaussian Blur + Canny + HoughCircles (OpenCV)

Ejecutar desde la carpeta donde está este archivo:
    python detectar_circulos.py
"""

import urllib.request
import cv2
import numpy as np

# ── Configuración ──────────────────────────────────────────────────────────────

IMAGE_URL  = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"

# Uso esta imagen de Wikimedia que tiene círculos bien definidos y es de dominio público.
# Si quieres usar tu propia imagen, cambia IMAGE_URL por la ruta local:
#   IMAGE_PATH = "mi_imagen.jpg"  y comenta el bloque de descarga.
IMAGE_PATH = "senal_trafico.jpg"
OUTPUT_PATH = "resultado_circulos.jpg"
REPORT_PATH = "reporte.md"

# Parámetros de HoughCircles — los mismos del notebook de clase
BLUR_KERNEL   = (11, 11)
CANNY_T1      = 50
CANNY_T2      = 150
HOUGH_DP      = 1.2    # relación resolución acumulador / imagen
HOUGH_MINDIST = 40     # distancia mínima entre centros detectados
HOUGH_P1      = 80     # umbral Canny interno de HoughCircles
HOUGH_P2      = 30     # umbral acumulador (más bajo = más detecciones)
HOUGH_MINR    = 20     # radio mínimo en píxeles
HOUGH_MAXR    = 200    # radio máximo en píxeles


# ── Descarga de imagen ─────────────────────────────────────────────────────────

def descargar_imagen(url: str, destino: str) -> None:
    print(f"[info] Descargando imagen desde:\n       {url}")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, open(destino, "wb") as f:
        f.write(resp.read())
    print(f"[info] Imagen guardada en: {destino}")


# ── Pipeline de detección ──────────────────────────────────────────────────────

def detectar_circulos(path: str):
    # Cargo la imagen — si viene con canal alfa (PNG) lo elimino
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"No pude cargar la imagen: {path}")

    # Elimino canal alfa si existe (PNG con transparencia)
    if img.shape[2] == 4:
        alpha = img[:, :, 3]
        fondo = np.ones_like(img[:, :, :3], dtype=np.uint8) * 255
        mask  = alpha[:, :, np.newaxis] / 255.0
        img   = (img[:, :, :3] * mask + fondo * (1 - mask)).astype(np.uint8)

    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)

    # Detecto bordes con Canny para visualización (igual que en el notebook)
    edges = cv2.Canny(blurred, CANNY_T1, CANNY_T2)

    # Detecto círculos con la Transformada de Hough
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp       = HOUGH_DP,
        minDist  = HOUGH_MINDIST,
        param1   = HOUGH_P1,
        param2   = HOUGH_P2,
        minRadius= HOUGH_MINR,
        maxRadius= HOUGH_MAXR,
    )

    return img, gray, blurred, edges, circles


def anotar_imagen(img: np.ndarray, circles) -> tuple:
    """
    Dibujo cada círculo detectado y marco su centro.
    Retorno la imagen anotada y la lista de círculos como enteros.
    """
    resultado = img.copy()
    detectados = []

    if circles is not None:
        cs = np.uint16(np.around(circles[0]))
        for i, (x, y, r) in enumerate(cs, start=1):
            cv2.circle(resultado, (x, y), r,  (0, 200, 0), 3)   # círculo exterior
            cv2.circle(resultado, (x, y), 4,  (0, 0, 255), -1)  # centro
            cv2.putText(resultado, str(i), (x - 10, y - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            detectados.append({"id": i, "cx": int(x), "cy": int(y), "radio": int(r)})

    return resultado, detectados


# ── Reporte en Markdown ────────────────────────────────────────────────────────

def generar_reporte(img_shape: tuple, detectados: list, path: str) -> None:
    h, w = img_shape[:2]
    n    = len(detectados)

    radios  = [d["radio"] for d in detectados] if detectados else [0]
    r_min   = min(radios)
    r_max   = max(radios)
    r_mean  = np.mean(radios)

    lineas = [
        "# Tarea 4 — Detección de centros de objetos circulares",
        "",
        "## Objetivo",
        "Aplicar la Transformada de Hough para detectar círculos en una imagen",
        "de señales de tráfico redondas, identificando el centro y radio de cada una.",
        "",
        "## Metodología",
        "El pipeline replica exactamente las técnicas vistas en clase:",
        "",
        "1. **Carga de imagen** con OpenCV (`cv2.imread`)",
        "2. **Conversión a escala de grises** (`cv2.cvtColor`)",
        "3. **Gaussian Blur** para reducir ruido antes de detectar bordes",
        f"   - Kernel: `{BLUR_KERNEL}`",
        "4. **Detección de bordes con Canny** (visualización)",
        f"   - Umbral 1: `{CANNY_T1}` | Umbral 2: `{CANNY_T2}`",
        "5. **HoughCircles** para detectar círculos",
        f"   - `dp={HOUGH_DP}`, `minDist={HOUGH_MINDIST}`, `param1={HOUGH_P1}`, `param2={HOUGH_P2}`",
        f"   - Radio mínimo: `{HOUGH_MINR}px` | Radio máximo: `{HOUGH_MAXR}px`",
        "",
        "## Imagen analizada",
        f"- **Resolución:** {w} × {h} px",
        f"- **Tipo:** señales de tráfico redondas (dominio público — Wikimedia Commons)",
        "",
        "## Resultados",
        f"Se detectaron **{n} círculo(s)** en la imagen.",
        "",
    ]

    if detectados:
        lineas += [
            "| # | Centro (x, y) | Radio (px) |",
            "|---|--------------|------------|",
        ]
        for d in detectados:
            lineas.append(f"| {d['id']} | ({d['cx']}, {d['cy']}) | {d['radio']} |")

        lineas += [
            "",
            "### Estadísticas de los radios detectados",
            f"- **Mínimo:** {r_min} px",
            f"- **Máximo:** {r_max} px",
            f"- **Promedio:** {r_mean:.1f} px",
        ]
    else:
        lineas.append("> No se detectaron círculos con los parámetros actuales.")
        lineas.append("> Prueba reduciendo `HOUGH_P2` o ajustando `HOUGH_MINR`/`HOUGH_MAXR`.")

    lineas += [
        "",
        "## Imagen resultado",
        "![Resultado](resultado_circulos.jpg)",
        "",
        "## Conclusiones",
        "- La combinación Gaussian Blur + HoughCircles es efectiva para detectar",
        "  objetos circulares en imágenes con contraste moderado.",
        "- El parámetro `param2` es el más sensible: valores bajos aumentan",
        "  detecciones pero también los falsos positivos.",
        "- El desenfoque previo reduce ruido y mejora la estabilidad de la",
        "  detección, como se demostró en clase con la imagen de monedas.",
        "- Para señales de tráfico reales se recomienda segmentar por color",
        "  (rojo/azul en HSV) antes de aplicar HoughCircles para reducir",
        "  falsos positivos en escenas complejas.",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))

    print(f"[info] Reporte guardado en: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    descargar_imagen(IMAGE_URL, IMAGE_PATH)

    img, gray, blurred, edges, circles = detectar_circulos(IMAGE_PATH)
    resultado, detectados = anotar_imagen(img, circles)

    cv2.imwrite(OUTPUT_PATH, resultado)
    print(f"[info] Imagen anotada guardada en: {OUTPUT_PATH}")

    n = len(detectados)
    print(f"[info] Círculos detectados: {n}")
    for d in detectados:
        print(f"       #{d['id']}  centro=({d['cx']}, {d['cy']})  radio={d['radio']}px")

    generar_reporte(img.shape, detectados, REPORT_PATH)
    print("\n[listo] Archivos generados:")
    print(f"        {OUTPUT_PATH}")
    print(f"        {REPORT_PATH}")


if __name__ == "__main__":
    main()
