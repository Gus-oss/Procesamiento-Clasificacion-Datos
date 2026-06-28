"""
Entrenamiento con early stopping, ReduceLROnPlateau y búsqueda de
hiperparámetros con Optuna (pruning bayesiano incluido).

Para EfficientNetB0 aplico entrenamiento en dos fases automáticamente:
  Fase 1 (épocas 0 … EPOCHS//2)  → backbone congelado
  Fase 2 (épocas EPOCHS//2 … N)  → últimos 3 bloques descongelados
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config.tarea4_config as cfg
from processors.tarea4.models import build_model, EfficientNetB0

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA = True
except ImportError:
    _OPTUNA = False
    print("[aviso] Optuna no instalado — uso hiperparámetros por defecto.")


# ── Utilidades ─────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        print(f"[info] GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("[info] Apple MPS")
        return torch.device("mps")
    print("[info] CPU")
    return torch.device("cpu")


def _make_optimizer(model: nn.Module, name: str, lr: float) -> torch.optim.Optimizer:
    # EfficientNetB0 en Fase 2 usa lr diferencial por grupos de parámetros.
    params = (
        model.param_groups(lr)
        if isinstance(model, EfficientNetB0)
        else model.parameters()
    )
    name = name.lower()
    if name == "adam":   return Adam(params,  lr=lr, weight_decay=1e-4)
    if name == "adamw":  return AdamW(params, lr=lr, weight_decay=1e-4)
    if name == "sgd":    return SGD(params,   lr=lr, weight_decay=1e-4, momentum=0.9)
    raise ValueError(f"Optimizador desconocido: {name}")


# ── Época de entrenamiento / evaluación ───────────────────────────────────────

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[Any],
) -> Tuple[float, float]:
    model.train()
    loss_sum, correct, n = 0.0, 0, 0

    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        loss_sum += loss.item() * imgs.size(0)
        correct  += (logits.argmax(1) == labels).sum().item()
        n        += imgs.size(0)

    return loss_sum / n, correct / n


@torch.no_grad()
def _eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0

    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss_sum += criterion(logits, labels).item() * imgs.size(0)
        correct  += (logits.argmax(1) == labels).sum().item()
        n        += imgs.size(0)

    return loss_sum / n, correct / n


# ── Entrenamiento completo ─────────────────────────────────────────────────────

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    hp:           Dict,
    device:       torch.device,
    ckpt_path:    str,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Entreno el modelo con early stopping y ReduceLROnPlateau.
    Para EfficientNetB0 la transición de Fase 1 a Fase 2 ocurre
    automáticamente a mitad del número de épocas configurado.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimizer = _make_optimizer(model, hp["optimizer"], hp["lr"])
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=cfg.LR_FACTOR, patience=cfg.LR_PATIENCE)
    scaler    = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_val_loss  = float("inf")
    patience_count = 0
    phase2_epoch   = cfg.EPOCHS // 2
    history        = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    print(f"\n{'─'*60}")
    print(f"Entrenando: lr={hp['lr']:.2e}  opt={hp['optimizer']}  dropout={hp['dropout']:.2f}")
    print(f"{'─'*60}")
    t0 = time.time()

    for epoch in range(cfg.EPOCHS):

        # Transición a Fase 2 para EfficientNetB0
        if isinstance(model, EfficientNetB0) and epoch == phase2_epoch:
            model.unfreeze_top(3)
            optimizer = _make_optimizer(model, hp["optimizer"], hp["lr"] / 10)
            scheduler = ReduceLROnPlateau(optimizer, "min", factor=cfg.LR_FACTOR, patience=cfg.LR_PATIENCE)

        tr_loss, tr_acc = _train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        vl_loss, vl_acc = _eval_epoch(model, val_loader, criterion, device)
        scheduler.step(vl_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss);   history["val_acc"].append(vl_acc)
        history["lr"].append(lr_now)

        print(f"  época {epoch+1:3d}/{cfg.EPOCHS}  "
              f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}  "
              f"vl_loss={vl_loss:.4f}  vl_acc={vl_acc:.4f}  lr={lr_now:.2e}")

        if vl_loss < best_val_loss:
            best_val_loss  = vl_loss
            patience_count = 0
            torch.save({"epoch": epoch + 1, "state": model.state_dict(),
                        "hp": hp, "val_loss": vl_loss}, ckpt_path)
        else:
            patience_count += 1
            if patience_count >= cfg.EARLY_STOP_PATIENCE:
                print(f"  [early stop] época {epoch+1}  mejor val_loss={best_val_loss:.4f}")
                break

    print(f"[info] Listo en {(time.time()-t0)/60:.1f} min  →  {ckpt_path}")
    return history


# ── Optuna ─────────────────────────────────────────────────────────────────────

def _objective(
    trial,
    model_name:    str,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    device:        torch.device,
    input_dim:     Optional[int],
    class_weights: Optional[torch.Tensor],
) -> float:
    lr      = trial.suggest_float("lr",      *cfg.LR_RANGE,      log=True)
    dropout = trial.suggest_float("dropout", *cfg.DROPOUT_RANGE)
    opt     = trial.suggest_categorical("optimizer", cfg.OPTIMIZERS)

    model = build_model(model_name, input_dim=input_dim, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimizer = _make_optimizer(model, opt, lr)
    best = float("inf")

    for epoch in range(cfg.TRIAL_EPOCHS):
        _train_epoch(model, train_loader, optimizer, criterion, device, None)
        vl_loss, _ = _eval_epoch(model, val_loader, criterion, device)
        trial.report(vl_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        best = min(best, vl_loss)

    return best


def search_hyperparams(
    model_name:    str,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    device:        torch.device,
    input_dim:     Optional[int]         = None,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Busco los mejores hiperparámetros con Optuna (pruning con MedianPruner).
    Si Optuna no está instalado devuelvo los valores por defecto del config.
    """
    if not _OPTUNA:
        return {"lr": cfg.LR, "dropout": 0.3, "optimizer": "AdamW"}

    print(f"\n[info] Búsqueda de hiperparámetros para '{model_name}' ({cfg.N_TRIALS} trials)…")
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    study.optimize(
        lambda t: _objective(t, model_name, train_loader, val_loader,
                             device, input_dim, class_weights),
        n_trials=cfg.N_TRIALS,
        show_progress_bar=True,
    )
    best = study.best_params
    print(f"[info] Mejores params: {best}  (val_loss={study.best_value:.4f})")
    return best


# ── Pipeline completo por modelo ───────────────────────────────────────────────

def train_pipeline(
    model_name:    str,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    device:        torch.device,
    input_dim:     Optional[int]         = None,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[nn.Module, Dict]:
    """
    Ejecuto el pipeline completo:
      1. Búsqueda de hiperparámetros con Optuna
      2. Entrenamiento final con los mejores hiperparámetros
      3. Cargo el mejor checkpoint antes de retornar
    """
    out = Path(cfg.OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    ckpt    = str(out / f"{model_name}_best.pth")
    hist_f  = str(out / f"{model_name}_history.json")

    hp    = search_hyperparams(model_name, train_loader, val_loader, device, input_dim, class_weights)
    model = build_model(model_name, input_dim=input_dim, dropout=hp.get("dropout", 0.3))

    history = train(model, train_loader, val_loader, hp, device, ckpt, class_weights)

    with open(hist_f, "w") as f:
        json.dump(history, f, indent=2)

    ckpt_data = torch.load(ckpt, map_location=device)
    model.load_state_dict(ckpt_data["state"])
    return model.to(device), history
