"""
Tarea Clase 5 — Redes Neuronales Convolucionales
Clasificacion de frutas con CNN, Transfer Learning y Fine-tuning
Dataset: Fruits-360 (Kaggle / kagglehub)

Comparativa:
  Modelo 1: CNN entrenada desde cero
  Modelo 2: VGG16 con Transfer Learning (capas congeladas)
  Modelo 3: ResNet50 con Fine-tuning (capas superiores descongeladas)

Mejoras respecto a v1:
  - 30 clases (antes 10) para mayor dificultad y validez del experimento
  - Epochs aumentados a 30 con patience=7 para que ResNet50 converja
  - Data augmentation mas agresiva para reducir overfitting de CNN Scratch
  - Matriz de confusion por modelo
  - Reporte de precision, recall y F1 por clase
  - Curvas de aprendizaje individuales por modelo
  - Seleccion de clases distribuida (no solo las primeras alfabeticamente)
  - Ruta de subset en directorio temporal de Windows
"""

import os
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, Input
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ─────────────────────────────────────────────
# CONFIGURACION GLOBAL
# ─────────────────────────────────────────────
IMG_SIZE    = (100, 100)
BATCH_SIZE  = 32
EPOCHS      = 30        # mas epochs para que fine-tuning converja
N_CLASES    = 30        # mas clases = problema mas dificil y realista
SEED        = 42
SUBSET_DIR  = os.path.join(os.environ.get("TEMP", "/tmp"), "fruits_subset")
OUTPUT_DIR  = r"C:\Users\PC\Documents\DocumentosGustavo\Github\Maestria\Procesamiento-Clasificacion-Datos\Procesamiento-Clasificacion-Datos\data\processed\tarea5"

tf.random.set_seed(SEED)
np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────

def descargar_dataset() -> tuple[str, str]:
    """Descarga Fruits-360 via kagglehub y localiza Training/ y Test/."""
    try:
        import kagglehub
    except ImportError:
        raise ImportError("Instala kagglehub: pip install kagglehub")

    path = kagglehub.dataset_download("moltean/fruits")
    print(f"Dataset en: {path}")

    for root, dirs, _ in os.walk(path):
        if "Training" in dirs and "Test" in dirs:
            return os.path.join(root, "Training"), os.path.join(root, "Test")

    raise FileNotFoundError(f"No se encontraron Training/Test en {path}")


def seleccionar_clases(train_dir: str, test_dir: str) -> tuple[str, str, list[str]]:
    """
    Selecciona N_CLASES distribuidas uniformemente del dataset
    y las copia a un directorio temporal para los generadores.
    Distribucion uniforme evita sesgos de solo tomar las primeras clases.
    """
    todas = sorted(os.listdir(train_dir))
    indices = np.linspace(0, len(todas) - 1, N_CLASES, dtype=int)
    clases = [todas[i] for i in indices]

    print(f"\nClases seleccionadas ({N_CLASES}):")
    for i, c in enumerate(clases):
        print(f"  {i:2d}. {c}")

    for split, src in [("train", train_dir), ("test", test_dir)]:
        dest = os.path.join(SUBSET_DIR, split)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        for clase in clases:
            src_cls = os.path.join(src, clase)
            if os.path.exists(src_cls):
                shutil.copytree(src_cls, os.path.join(dest, clase))

    return os.path.join(SUBSET_DIR, "train"), os.path.join(SUBSET_DIR, "test"), clases


# ─────────────────────────────────────────────
# 2. GENERADORES DE DATOS
# ─────────────────────────────────────────────

def crear_generadores(train_dir: str, test_dir: str):
    """
    Data augmentation mas agresiva para CNN Scratch para reducir overfitting.
    VGG16 y ResNet50 usan sus propias funciones de preprocesamiento.
    """
    aug_train = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.15,
        shear_range=0.1,
        brightness_range=[0.8, 1.2],
        validation_split=0.15,
    )
    solo_rescale = ImageDataGenerator(rescale=1.0 / 255)

    kwargs_comun = dict(
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        seed=SEED,
    )

    train_data = aug_train.flow_from_directory(train_dir, subset="training",   **kwargs_comun)
    val_data   = aug_train.flow_from_directory(train_dir, subset="validation", **kwargs_comun)
    test_data  = solo_rescale.flow_from_directory(test_dir, shuffle=False,     **kwargs_comun)

    return train_data, val_data, test_data, len(train_data.class_indices)


# ─────────────────────────────────────────────
# 3. MODELOS
# ─────────────────────────────────────────────

def modelo_cnn_scratch(n_clases: int) -> models.Model:
    """
    CNN desde cero con 4 bloques Conv+BN+Pool.
    L2 regularization en capas densas para combatir overfitting.
    """
    from tensorflow.keras.regularizers import l2

    inputs = Input(shape=(*IMG_SIZE, 3))
    x = inputs

    for filtros in [32, 64, 128, 256]:
        x = layers.Conv2D(filtros, (3, 3), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filtros, (3, 3), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_clases, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="CNN_Scratch")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def modelo_transfer_vgg16(n_clases: int) -> models.Model:
    """
    Transfer Learning con VGG16. Capas base congeladas.
    Se agrega BatchNormalization antes de la capa densa para estabilizar.
    """
    base = VGG16(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    base.trainable = False

    inputs = Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_clases, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="VGG16_Transfer")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def modelo_finetuning_resnet50(n_clases: int) -> models.Model:
    """
    Fine-tuning con ResNet50. Se descongelan las ultimas 50 capas
    (antes eran 20, insuficiente para adaptar features de alto nivel).
    Learning rate de 5e-5 para no destruir pesos preentrenados.
    """
    base = ResNet50(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))

    for layer in base.layers[:-50]:
        layer.trainable = False
    for layer in base.layers[-50:]:
        layer.trainable = True

    inputs = Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_clases, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="ResNet50_FineTuning")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────
# 4. ENTRENAMIENTO
# ─────────────────────────────────────────────

def entrenar(model, train_data, val_data, nombre: str):
    """Entrena con EarlyStopping (patience=7) y ReduceLROnPlateau."""
    cb = [
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=7,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=4, min_lr=1e-7, verbose=1,
        ),
    ]

    print(f"\n{'='*52}\n Entrenando: {nombre}\n{'='*52}")
    t0 = time.time()
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=cb,
        verbose=1,
    )
    elapsed = time.time() - t0
    print(f"Tiempo: {elapsed / 60:.1f} min")
    return history, elapsed


# ─────────────────────────────────────────────
# 5. EVALUACION
# ─────────────────────────────────────────────

def evaluar_modelo(model, test_data, clases: list[str]) -> dict:
    """
    Retorna accuracy, predicciones y etiquetas reales.
    Necesario para matriz de confusion y reporte por clase.
    """
    test_data.reset()
    loss, acc = model.evaluate(test_data, verbose=0)

    test_data.reset()
    y_pred_prob = model.predict(test_data, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_real = test_data.classes

    return {"loss": loss, "accuracy": acc, "y_pred": y_pred, "y_real": y_real}


# ─────────────────────────────────────────────
# 6. VISUALIZACIONES
# ─────────────────────────────────────────────

def graficar_curvas(history, nombre: str, ruta: str):
    """Curvas de accuracy y loss individuales por modelo."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epocas = range(1, len(history.history["accuracy"]) + 1)

    ax1.plot(epocas, history.history["accuracy"],     color="#378ADD", label="Train")
    ax1.plot(epocas, history.history["val_accuracy"], color="#378ADD", alpha=0.5,
             linestyle="--", label="Validacion")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoca")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(epocas, history.history["loss"],     color="#E24B4A", label="Train")
    ax2.plot(epocas, history.history["val_loss"], color="#E24B4A", alpha=0.5,
             linestyle="--", label="Validacion")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoca")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.suptitle(f"Curvas de entrenamiento — {nombre}", fontsize=13)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Guardado: {ruta}")


def graficar_matriz_confusion(y_real, y_pred, clases: list[str], nombre: str, ruta: str):
    """Matriz de confusion normalizada."""
    cm = confusion_matrix(y_real, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=90, values_format=".2f")
    ax.set_title(f"Matriz de Confusion (normalizada) — {nombre}", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Guardado: {ruta}")


def graficar_comparativa(nombres: list, resultados: list[dict], tiempos: list, ruta: str):
    """Barras comparativas de accuracy y tiempo entre los tres modelos."""
    accuracies = [r["accuracy"] * 100 for r in resultados]
    colores = ["#378ADD", "#1D9E75", "#E24B4A"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, valores, ylabel, titulo, fmt in [
        (ax1, accuracies,           "Test Accuracy (%)",              "Accuracy",              "{:.1f}%"),
        (ax2, [t / 60 for t in tiempos], "Tiempo de entrenamiento (min)", "Tiempo entrenamiento", "{:.1f} min"),
    ]:
        bars = ax.bar(nombres, valores, color=colores, width=0.5, edgecolor="none")
        for bar, val in zip(bars, valores):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(valores) * 0.02,
                    fmt.format(val), ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_title(titulo)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Comparativa: CNN Scratch vs Transfer Learning vs Fine-tuning", fontsize=13)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Guardado: {ruta}")


def graficar_predicciones(model, test_data, clases: list[str], ruta: str, n: int = 16):
    """Cuadricula de imagenes con prediccion del mejor modelo."""
    test_data.reset()
    imgs, labels = next(test_data)
    preds = model.predict(imgs, verbose=0)

    n = min(n, len(imgs))
    cols, rows = 4, n // 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        ax.imshow(imgs[i])
        real = clases[np.argmax(labels[i])]
        pred = clases[np.argmax(preds[i])]
        conf = np.max(preds[i]) * 100
        ax.set_title(f"Real: {real}\nPred: {pred} ({conf:.0f}%)",
                     fontsize=8, color="green" if real == pred else "red")
        ax.axis("off")

    plt.suptitle("Predicciones del mejor modelo", fontsize=13)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Guardado: {ruta}")


def imprimir_reporte(nombre: str, resultado: dict, clases: list[str]):
    """Imprime precision, recall y F1 por clase."""
    print(f"\n{'─'*52}")
    print(f" Reporte de clasificacion — {nombre}")
    print(f"{'─'*52}")
    print(classification_report(resultado["y_real"], resultado["y_pred"],
                                 target_names=clases, digits=3))


def imprimir_resumen_final(nombres: list, resultados: list[dict], tiempos: list):
    print("\n" + "=" * 58)
    print(f" {'Modelo':<28} {'Accuracy':>10} {'Tiempo':>14}")
    print("-" * 58)
    for nombre, res, t in zip(nombres, resultados, tiempos):
        print(f" {nombre:<28} {res['accuracy']*100:>9.2f}% {t/60:>12.1f} min")
    print("=" * 58)
    mejor = nombres[np.argmax([r["accuracy"] for r in resultados])]
    print(f"\nMejor modelo: {mejor}")


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

def main():
    print("Clasificacion de Frutas con CNN")
    print("Procesamiento y Clasificacion de Datos — Clase 5\n")

    # Datos
    train_dir, test_dir = descargar_dataset()
    train_dir, test_dir, clases = seleccionar_clases(train_dir, test_dir)
    train_data, val_data, test_data, n_clases = crear_generadores(train_dir, test_dir)

    # Modelos a comparar
    modelos = [
        ("CNN Scratch",          modelo_cnn_scratch(n_clases)),
        ("VGG16 Transfer",       modelo_transfer_vgg16(n_clases)),
        ("ResNet50 Fine-tuning", modelo_finetuning_resnet50(n_clases)),
    ]

    historias, resultados, tiempos = [], [], []

    for nombre, model in modelos:
        hist, elapsed = entrenar(model, train_data, val_data, nombre)
        res = evaluar_modelo(model, test_data, clases)

        historias.append(hist)
        resultados.append(res)
        tiempos.append(elapsed)

        print(f"  Test Accuracy: {res['accuracy']*100:.2f}%")

        # Graficas individuales por modelo
        nombre_archivo = nombre.lower().replace(" ", "_")
        graficar_curvas(hist, nombre,
                        os.path.join(OUTPUT_DIR, f"curvas_{nombre_archivo}.png"))
        graficar_matriz_confusion(res["y_real"], res["y_pred"], clases,
                                   nombre,
                                   os.path.join(OUTPUT_DIR, f"confusion_{nombre_archivo}.png"))
        imprimir_reporte(nombre, res, clases)

    # Graficas comparativas
    nombres = [n for n, _ in modelos]
    graficar_comparativa(nombres, resultados, tiempos,
                         os.path.join(OUTPUT_DIR, "comparativa_modelos.png"))

    # Predicciones del mejor modelo
    mejor_idx = np.argmax([r["accuracy"] for r in resultados])
    test_data.reset()
    graficar_predicciones(modelos[mejor_idx][1], test_data, clases,
                          os.path.join(OUTPUT_DIR, "predicciones.png"))

    imprimir_resumen_final(nombres, resultados, tiempos)


if __name__ == "__main__":
    main()