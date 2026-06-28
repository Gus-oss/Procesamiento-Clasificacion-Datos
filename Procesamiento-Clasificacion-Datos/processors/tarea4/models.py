"""
Tres modelos de clasificación binaria (bueno / defectuoso) para MVTec AD.

  MLP            — red densa sobre el vector de features de OpenCV
  CustomCNN      — CNN desde cero con bloques residuales simples
  EfficientNetB0 — transfer learning con fine-tuning en dos fases
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ── MLP ────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """
    Perceptrón multicapa sobre el vector de features de OpenCV (~300 dims).
    Uso BatchNorm después de cada capa lineal para estabilizar el entrenamiento,
    ya que las features de Canny, Harris y LBP tienen escalas muy diferentes.
    """

    def __init__(self, input_dim: int, hidden: List[int] = None, dropout: float = 0.3):
        super().__init__()
        hidden = hidden or [512, 256, 128]
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── CustomCNN ─────────────────────────────────────────────────────────────────

class _ResBlock(nn.Module):
    """
    Bloque residual simple: dos convoluciones 3×3 con skip connection.
    El skip se proyecta con una conv 1×1 cuando cambian los canales o el stride.
    """

    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
            nn.BatchNorm2d(c_out),
        ) if stride != 1 or c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.body(x) + self.skip(x), inplace=True)


class CustomCNN(nn.Module):
    """
    CNN con cuatro stages de profundidad creciente (64→128→256→512 canales).
    Diseñada para imágenes 224×224; el GlobalAvgPool al final la hace
    independiente del tamaño de entrada.
    """

    def __init__(self, dropout: float = 0.4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.stages = nn.Sequential(
            _ResBlock(64, 64),   _ResBlock(64, 64),
            _ResBlock(64, 128, stride=2),  _ResBlock(128, 128),
            _ResBlock(128, 256, stride=2), _ResBlock(256, 256),
            _ResBlock(256, 512, stride=2), _ResBlock(512, 512),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 2),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stages(self.stem(x)))


# ── EfficientNet-B0 con fine-tuning en dos fases ──────────────────────────────

class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 preentrenado en ImageNet con una cabeza nueva de 2 clases.

    Fase 1 (feature extraction): backbone congelado, solo entreno la cabeza.
    Fase 2 (fine-tuning):        descongelo los últimos bloques con lr/10.

    Esta estrategia evita el catastrophic forgetting y es especialmente útil
    cuando el dataset de fine-tuning es pequeño en relación a ImageNet.
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512), nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, 2),
        )
        self.freeze_backbone()

    def freeze_backbone(self):
        for name, p in self.backbone.named_parameters():
            p.requires_grad = "classifier" in name
        self._log_trainable("Fase 1 — backbone congelado")

    def unfreeze_top(self, n_blocks: int = 3):
        """Descongelo los últimos n_blocks del backbone para fine-tuning."""
        blocks = list(self.backbone.features.children())
        for block in blocks[-n_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
        self._log_trainable(f"Fase 2 — últimos {n_blocks} bloques descongelados")

    def param_groups(self, lr: float) -> List[dict]:
        """
        Retorno grupos de parámetros con learning rate diferencial:
        el backbone recibe lr/10 para no borrar los pesos preentrenados.
        """
        backbone_p   = [p for n, p in self.backbone.named_parameters()
                        if "classifier" not in n and p.requires_grad]
        classifier_p = [p for n, p in self.backbone.named_parameters()
                        if "classifier" in n and p.requires_grad]
        return [
            {"params": backbone_p,   "lr": lr / 10},
            {"params": classifier_p, "lr": lr},
        ]

    def _log_trainable(self, label: str):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"[info] {label}: {trainable:,} / {total:,} parámetros entrenables")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(name: str, input_dim: Optional[int] = None, dropout: float = 0.3) -> nn.Module:
    """
    Instancio un modelo por nombre para no acoplar train.py a las clases.

    Args:
        name:      "mlp" | "cnn" | "efficientnet"
        input_dim: requerido solo para "mlp"
        dropout:   tasa de dropout
    """
    name = name.lower()
    if name == "mlp":
        if input_dim is None:
            raise ValueError("input_dim es obligatorio para MLP")
        return MLP(input_dim, dropout=dropout)
    if name == "cnn":
        return CustomCNN(dropout=dropout)
    if name == "efficientnet":
        return EfficientNetB0(dropout=dropout)
    raise ValueError(f"Modelo desconocido: '{name}'. Opciones: mlp | cnn | efficientnet")
