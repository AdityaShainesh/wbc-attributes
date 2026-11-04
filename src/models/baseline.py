"""The baseline: a plain CNN predicting the 5-part differential directly from pixels.

This is the control arm. The research question is whether forcing predictions through an
11-attribute bottleneck generalizes better across labs, and that claim only means
something relative to an unconstrained CNN trained under an identical protocol. So this
model deliberately does nothing clever: no attribute supervision, no stain
normalization, no domain-adaptation trick.

Keep it that way. Any improvement added here has to be added to the bottleneck arm too,
or the comparison stops being a comparison.
"""
import torch.nn as nn
from torchvision import models

ARCHITECTURES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
}


class BaselineCNN(nn.Module):
    """ImageNet-pretrained backbone with the classifier replaced by a 5-way head."""

    def __init__(self, arch="resnet18", pretrained=True, num_classes=5, dropout=0.0):
        super().__init__()
        if arch not in ARCHITECTURES:
            raise ValueError(f"unknown arch {arch!r}; known: {', '.join(ARCHITECTURES)}")
        constructor, weights = ARCHITECTURES[arch]
        self.arch = arch
        self.pretrained = pretrained
        self.backbone = constructor(weights=weights if pretrained else None)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = (
            nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
            if dropout else nn.Linear(in_features, num_classes)
        )
        self.num_classes = num_classes

    def forward(self, images):
        return self.backbone(images)

    def describe(self):
        """Recorded in the results file, so a run's architecture is readable from it."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "kind": "baseline_cnn",
            "arch": self.arch,
            "pretrained": self.pretrained,
            "num_classes": self.num_classes,
            "trainable_parameters": trainable,
        }


def build_model(config):
    """Build the model named by config["model"]["kind"]."""
    spec = dict(config["model"])
    kind = spec.pop("kind", "baseline_cnn")
    if kind != "baseline_cnn":
        raise ValueError(
            f"unknown model kind {kind!r}; this session only implements 'baseline_cnn' "
            f"(the attribute-bottleneck model is a later session)"
        )
    return BaselineCNN(**spec)
