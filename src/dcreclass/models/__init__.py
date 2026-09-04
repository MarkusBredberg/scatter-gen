# models/__init__.py
# Expose model classes for import by training/evaluation scripts.
# Usage: from dcreclass.models import DualScatterSqueezeNet

from .classifiers import ImageCNN, ScatterNet, SimpleScatterNet, DualCNNSqueezeNet, DualScatterSqueezeNet