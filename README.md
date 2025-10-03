## Objective, Target, and Analysis

This project trains a compact depthwise-separable CNN (MobileNet-inspired) on CIFAR-10 aiming for strong accuracy with low params and good inference efficiency. We iterated on augmentation, dropout, scheduler, and the network layout to steadily improve generalization.

Brief Methodology:
- Start from a simple skeleton, then add augmentations, dropout, and stride — measure RF each step.
- Reduce parameters using depthwise separable and dilated convolutions where appropriate; verify padding/stride choices.
- Iterate one change at a time; validate hyperparameters. Scheduler choices matter.

## Logs Summary (Train/Test Accuracy)

Source logs: `logs/model_accuracy.log` (full history). Selected checkpoints:

| Epoch | LR       | Train Acc (%) | Test Acc (%) |
|------:|----------|---------------:|-------------:|
| 0     | 0.010000 | 33.25 | 44.40 |
| 10    | 0.009619 | 71.84 | 75.30 |
| 20    | 0.008536 | 77.57 | 81.73 |
| 30    | 0.006914 | 80.96 | 83.97 |
| 40    | 0.005001 | 83.18 | 84.77 |
| 50    | 0.003087 | 85.19 | 86.07 |
| 60    | 0.001465 | 86.37 | 86.55 |
| 70    | 0.000382 | 87.10 | 87.25 |
| 79    | 0.000005 | 87.45 | 87.20 |

Train accuracy rises steadily in tandem; see full per-epoch details in the log.

## Model Parameters, RF, and Output Size

Torch summary (`logs/torch_summary.log`):

```1:12:/Users/namratan/TSAI/ERA V4/Session 7/logs/torch_summary.log
cuda
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             864
       BatchNorm2d-2           [-1, 32, 32, 32]              64
              ReLU-3           [-1, 32, 32, 32]               0
            Conv2d-4           [-1, 32, 32, 32]             288
            Conv2d-5           [-1, 32, 32, 32]           1,024
```

Total params: 155,434. Layers in `model.py` include comments with RF and output sizes for each stage

## Model Architecture

- Input block
  - Conv: 3 → 32, kernel=3, stride=1, padding=1 (standard conv)
  - BN + ReLU
  - Output: 32×32; RF≈3
  - Summary: lightweight stem to extract basic edges/colors while preserving spatial resolution.

- ConvBlock1 (depthwise-separable)
  - Depthwise: kernel=3, stride=1, padding=1, groups=32
  - Pointwise: kernel=1, stride=1 → 32 ch; BN + ReLU
  - Depthwise: kernel=3, stride=2, padding=1, groups=32 (downsample 32→16)
  - Pointwise: kernel=1, stride=1 → 64 ch; BN + ReLU
  - Output: 16×16; RF≈5→7 across block; uses depthwise, no dilation
  - Summary: efficient feature extraction and first spatial downsampling using depthwise+pointwise.

- ConvBlock2 (depthwise-separable)
  - Depthwise: kernel=3, stride=1, padding=1, groups=64
  - Pointwise: kernel=1, stride=1 → 64 ch; BN + ReLU
  - Depthwise: kernel=3, stride=2, padding=1, groups=64 (downsample 16→8)
  - Pointwise: kernel=1, stride=1 → 128 ch; BN + ReLU
  - Output: 8×8; RF≈11→15; uses depthwise, no dilation
  - Summary: increases channel capacity while reducing resolution to capture mid-level patterns.

- ConvBlock3 (depthwise-separable)
  - Depthwise: kernel=3, stride=1, padding=1, groups=128
  - Pointwise: kernel=1, stride=1 → 128 ch; BN + ReLU
  - Depthwise: kernel=3, stride=2, padding=1, groups=128 (downsample 8→4)
  - Pointwise: kernel=1, stride=1 → 192 ch; BN + ReLU
  - Output: 4×4; RF≈23→31; uses depthwise, no dilation
  - Summary: compacts spatial map to 4×4 and enriches semantics for high-level features.

- ConvBlock4 (depthwise-separable + dilation)
  - Depthwise: kernel=3, stride=1, padding=1, groups=192
  - Pointwise: kernel=1, stride=1 → 192 ch; BN + ReLU
  - Depthwise (dilated): kernel=3, dilation=2, stride=1, padding=2, groups=192
  - Pointwise: kernel=1, stride=1 → 256 ch; BN + ReLU
  - Output: 4×4; RF≈47→79; uses depthwise and dilated depthwise conv
  - Summary: expands receptive field without further downsampling to aggregate broader context.

- Head
  - GAP: AdaptiveAvgPool2d → 1×1 (RF=full image)
  - Classifier: Conv 1×1, 256 → 10
  - Summary: global pooling and linear classifier for class logits.


## Components in `model.py`

- Augmentations: Albumentations (flip, shift/scale/rotate, coarse dropout) + normalization. Test transform keeps the wrapper empty for clarity but applies only normalization.
- Architecture: Depthwise separable stacks with occasional stride-2 for downsampling. Dilation used in the last block to expand RF without further spatial reduction. BatchNorm + ReLU after each block. GAP + 1x1 conv for classification.

Intuition: preserve early spatial detail, reduce resolution in controlled steps, increase channels deeper, and widen receptive field efficiently via depthwise/dilated ops.

## Folder Structure

```
Session 7/
  config.py        # Centralized hyperparameters and dataloader settings
  main.py          # Training entry point; wires datasets, loaders, model, optimizer, scheduler
  model.py         # Transforms, datasets, dataloaders, Net, train/test utilities
  logs/
    model_accuracy.log  # Per-epoch train/test metrics
    torch_summary.log   # Model summary and parameter counts
  S7_Final.ipynb   # Original notebook source
  My Learnings.md  # Iteration notes and insights
  README.md        # This file
```

To run training:

```bash
python "/Users/namratan/TSAI/ERA V4/Session 7/main.py"
```


