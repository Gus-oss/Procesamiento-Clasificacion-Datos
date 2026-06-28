"""
Tarea Clase 5 — Redes Neuronales Convolucionales
Clasificacion de frutas con CNN, Transfer Learning y Fine-tuning
Dataset: Fruits-360 (Kaggle / kagglehub)

Comparativa:
  Modelo 1: CNN entrenada desde cero
  Modelo 2: VGG16 con Transfer Learning (capas congeladas)
  Modelo 3: ResNet50 con Fine-tuning (capas superiores descongeladas)
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─────────────────────────────────────────────
# CONFIGURACION
# ─────────────────────────────────────────────
IMG_SIZE    = (100, 100)
BATCH_SIZE  = 32
EPOCHS      = 15
N_CLASES    = 10       # frutas a usar (subset para entrenamiento rapido)
SEED        = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# 1. DESCARGA DEL DATASET
# ─────────────────────────────────────────────

def descargar_dataset():
    """
    Descarga Fruits-360 desde Kaggle usando kagglehub.
    Requiere tener configuradas las credenciales de Kaggle:
        ~/.kaggle/kaggle.json  o  variables KAGGLE_USERNAME / KAGGLE_KEY
    Alternativa manual: descargar desde
        https://www.kaggle.com/datasets/moltean/fruits
    """
    try:
        import kagglehub
        path = kagglehub.dataset_download("moltean/fruits")
        # El dataset tiene subcarpetas Training/ y Test/
        train_dir = os.path.join(path, "fruits-360_dataset", "fruits-360", "Training")
        test_dir  = os.path.join(path, "fruits-360_dataset", "fruits-360", "Test")
        if not os.path.exists(train_dir):
            # Estructura alternativa segun version del dataset
            train_dir = os.path.join(path, "Training")
            test_dir  = os.path.join(path, "Test")
        print(f"Dataset descargado en: {path}")
        return train_dir, test_dir
    except ImportError:
        print("Instala kagglehub: pip install kagglehub")
        raise
    except Exception as e:
        print(f"Error al descargar: {e}")
        print("Descarga manual: https://www.kaggle.com/datasets/moltean/fruits")
        raise


def seleccionar_clases(train_dir, test_dir, n_clases=N_CLASES):
    """
    Selecciona las primeras n_clases carpetas (por orden alfabetico)
    para mantener el entrenamiento manejable.
    Crea symlinks temporales en subdirectorios filtrados.
    """
    import shutil

    clases = sorted(os.listdir(train_dir))[:n_clases]
    print(f"\nClases seleccionadas ({n_clases}):")
    for i, c in enumerate(clases):
        print(f"  {i:2d}. {c}")

    base = "/tmp/fruits_subset"
    for split, src in [("train", train_dir), ("test", test_dir)]:
        dest = os.path.join(base, split)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        for clase in clases:
            src_cls  = os.path.join(src, clase)
            dest_cls = os.path.join(dest, clase)
            if os.path.exists(src_cls):
                shutil.copytree(src_cls, dest_cls)

    return os.path.join(base, "train"), os.path.join(base, "test"), clases


# ─────────────────────────────────────────────
# 2. GENERADORES DE DATOS
# ─────────────────────────────────────────────

def crear_generadores(train_dir, test_dir):
    # Data augmentation solo en entrenamiento
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        validation_split=0.15,
    )
    test_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        seed=SEED,
    )
    val_data = train_gen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        seed=SEED,
    )
    test_data = test_gen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    n_clases = len(train_data.class_indices)
    return train_data, val_data, test_data, n_clases


# ─────────────────────────────────────────────
# 3. DEFINICION DE MODELOS
# ─────────────────────────────────────────────

def modelo_cnn_scratch(n_clases, input_shape=(*IMG_SIZE, 3)):
    """
    CNN entrenada desde cero.
    Arquitectura simple: 3 bloques Conv+Pool + 2 capas densas.
    """
    model = models.Sequential([
        # Bloque 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Bloque 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Bloque 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Clasificador
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(n_clases, activation="softmax"),
    ], name="CNN_Scratch")

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def modelo_transfer_vgg16(n_clases, input_shape=(*IMG_SIZE, 3)):
    """
    Transfer Learning con VGG16 preentrenado en ImageNet.
    Capas base completamente congeladas.
    """
    base = VGG16(weights="imagenet", include_top=False,
                 input_shape=input_shape)
    base.trainable = False          # congelar todo el modelo base

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(n_clases, activation="softmax"),
    ], name="VGG16_TransferLearning")

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def modelo_finetuning_resnet50(n_clases, input_shape=(*IMG_SIZE, 3)):
    """
    Fine-tuning con ResNet50.
    Las ultimas 20 capas se descongelan para ajuste fino.
    Se usa una tasa de aprendizaje menor para no destruir los pesos preentrenados.
    """
    base = ResNet50(weights="imagenet", include_top=False,
                    input_shape=input_shape)

    # Congelar todas las capas excepto las ultimas 20
    for layer in base.layers[:-20]:
        layer.trainable = False
    for layer in base.layers[-20:]:
        layer.trainable = True

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(n_clases, activation="softmax"),
    ], name="ResNet50_FineTuning")

    # Tasa de aprendizaje baja para fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────
# 4. ENTRENAMIENTO
# ─────────────────────────────────────────────

def entrenar(model, train_data, val_data, epochs=EPOCHS, nombre="modelo"):
    cb = [
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, min_lr=1e-6, verbose=1,
        ),
    ]

    print(f"\n{'='*50}")
    print(f" Entrenando: {nombre}")
    print(f"{'='*50}")
    t0 = time.time()

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=cb,
        verbose=1,
    )

    elapsed = time.time() - t0
    print(f"\nTiempo de entrenamiento: {elapsed/60:.1f} min")
    return history, elapsed


# ─────────────────────────────────────────────
# 5. EVALUACION Y VISUALIZACION
# ─────────────────────────────────────────────

def evaluar(model, test_data):
    loss, acc = model.evaluate(test_data, verbose=0)
    return loss, acc


def graficar_historia(historias, nombres, ruta="curvas_entrenamiento.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colores = ["#378ADD", "#1D9E75", "#E24B4A"]
    estilos = ["-", "--", "-."]

    for ax, metrica, titulo in zip(axes, ["accuracy", "loss"],
                                    ["Accuracy", "Loss"]):
        for hist, nombre, color, estilo in zip(historias, nombres, colores, estilos):
            ep = range(1, len(hist.history[metrica]) + 1)
            ax.plot(ep, hist.history[metrica],
                    color=color, linestyle=estilo, linewidth=2,
                    label=f"{nombre} (train)")
            ax.plot(ep, hist.history[f"val_{metrica}"],
                    color=color, linestyle=estilo, linewidth=2,
                    alpha=0.45, label=f"{nombre} (val)")
        ax.set_title(titulo, fontsize=13, pad=10)
        ax.set_xlabel("Epoca")
        ax.set_ylabel(titulo)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)

    plt.suptitle("Comparativa de modelos — Fruits-360", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Grafica guardada: {ruta}")


def graficar_comparativa(nombres, accuracies, tiempos,
                          ruta="comparativa_modelos.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colores = ["#378ADD", "#1D9E75", "#E24B4A"]

    # Accuracy
    bars = ax1.bar(nombres, [a * 100 for a in accuracies],
                   color=colores, width=0.5, edgecolor="none")
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f"{acc*100:.1f}%", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("Accuracy por modelo")
    ax1.spines[["top","right"]].set_visible(False)
    ax1.grid(axis="y", alpha=0.3)

    # Tiempo de entrenamiento
    bars2 = ax2.bar(nombres, [t/60 for t in tiempos],
                    color=colores, width=0.5, edgecolor="none")
    for bar, t in zip(bars2, tiempos):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.1,
                 f"{t/60:.1f} min", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")
    ax2.set_ylabel("Tiempo de entrenamiento (min)")
    ax2.set_title("Tiempo de entrenamiento")
    ax2.spines[["top","right"]].set_visible(False)
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("Comparativa: CNN Scratch vs Transfer Learning vs Fine-tuning",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Grafica guardada: {ruta}")


def mostrar_predicciones(model, test_data, clases,
                          ruta="predicciones.png", n=12):
    """
    Muestra una cuadricula de imagenes de prueba con su prediccion.
    """
    imgs, labels = next(test_data)
    preds = model.predict(imgs, verbose=0)

    n = min(n, len(imgs))
    cols = 4
    rows = n // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        ax.imshow(imgs[i])
        real  = clases[np.argmax(labels[i])]
        pred  = clases[np.argmax(preds[i])]
        conf  = np.max(preds[i]) * 100
        color = "green" if real == pred else "red"
        ax.set_title(f"Real: {real}\nPred: {pred} ({conf:.0f}%)",
                     fontsize=8, color=color)
        ax.axis("off")

    plt.suptitle("Predicciones del mejor modelo en test", fontsize=13)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Grafica guardada: {ruta}")


def imprimir_resumen(nombres, accuracies, tiempos):
    print("\n" + "="*55)
    print(f" {'Modelo':<28} {'Accuracy':>10} {'Tiempo':>12}")
    print("-"*55)
    for nombre, acc, t in zip(nombres, accuracies, tiempos):
        print(f" {nombre:<28} {acc*100:>9.2f}% {t/60:>10.1f} min")
    print("="*55)

    mejor_idx = np.argmax(accuracies)
    print(f"\nMejor modelo: {nombres[mejor_idx]} "
          f"({accuracies[mejor_idx]*100:.2f}%)")


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def main():
    print("Clasificacion de Frutas con CNN")
    print("Procesamiento y Clasificacion de Datos — Clase 5\n")

    # --- Datos ---
    train_dir, test_dir = descargar_dataset()
    train_dir, test_dir, clases = seleccionar_clases(train_dir, test_dir)
    train_data, val_data, test_data, n_clases = crear_generadores(
        train_dir, test_dir
    )

    # --- Modelos ---
    modelos = [
        ("CNN Scratch",         modelo_cnn_scratch(n_clases)),
        ("VGG16 Transfer",      modelo_transfer_vgg16(n_clases)),
        ("ResNet50 Fine-tuning",modelo_finetuning_resnet50(n_clases)),
    ]

    historias, accuracies, tiempos = [], [], []

    for nombre, model in modelos:
        model.summary(print_fn=lambda x: None)     # suprimir salida verbosa
        hist, elapsed = entrenar(model, train_data, val_data,
                                  epochs=EPOCHS, nombre=nombre)
        _, acc = evaluar(model, test_data)
        historias.append(hist)
        accuracies.append(acc)
        tiempos.append(elapsed)
        print(f"  Test Accuracy: {acc*100:.2f}%")

    # --- Visualizaciones ---
    nombres = [n for n, _ in modelos]
    graficar_historia(historias, nombres)
    graficar_comparativa(nombres, accuracies, tiempos)

    # Predicciones del mejor modelo
    mejor_modelo = modelos[np.argmax(accuracies)][1]
    test_data.reset()
    mostrar_predicciones(mejor_modelo, test_data, clases)

    imprimir_resumen(nombres, accuracies, tiempos)


if __name__ == "__main__":
    main()
