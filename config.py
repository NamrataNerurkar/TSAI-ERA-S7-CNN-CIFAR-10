from __future__ import annotations

# Training configuration
SEED: int = 1
EPOCHS: int = 80

# Optimizer
LR: float = 0.01
MOMENTUM: float = 0.9
WEIGHT_DECAY: float = 1e-4

# Data
DATA_DIR: str = "./data"
DOWNLOAD: bool = True

# Dataloader
BATCH_SIZE_CUDA: int = 128
BATCH_SIZE_CPU: int = 64
NUM_WORKERS: int = 4
PIN_MEMORY: bool = True
SHUFFLE: bool = True


