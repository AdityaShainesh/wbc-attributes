"""Models. build_model(config) is the only entry point training and eval should use."""
from src.models.baseline import BaselineCNN, build_model

__all__ = ["build_model", "BaselineCNN"]
