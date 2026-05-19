# GitHub Maintenance Guide

## Scope of This Repository

This repository should be maintained as a clean, reproducible codebase for the MNIST CNN homework project.
Only source code, configurations, and lightweight documentation should be committed.

## What Should Be Committed

- source code under `data/`, `models/`, `engine/`, `optimizers/`, `utils/`, and `scripts/`
- experiment configs under `configs/`
- project documentation under `docs/`
- report sources in Markdown under `report/`
- placeholder files such as `.gitkeep`

## What Should Not Be Committed

- dataset files such as `data/MNIST/`
- model checkpoints such as `*.pth`, `*.pt`, `*.ckpt`
- generated training logs
- generated figures and prediction outputs
- local virtual environments
- IDE settings and notebook checkpoints
- temporary Office files and binary report exports such as `.docx`

## Data Policy

MNIST is downloaded locally at runtime through `torchvision.datasets.MNIST`.
The repository should contain code for downloading and preparing data, but not the dataset itself.

## Output Policy

All generated artifacts should remain local and live under `outputs/`.
This includes:

- checkpoints
- logs
- figures
- prediction outputs

These files are intentionally ignored by Git so the repository remains lightweight and reviewable.

## Recommended Commit Style

Prefer small, scoped commits with action-oriented messages, for example:

- `add cnn basic training config`
- `fix augmentation dataset split bug`
- `add residual cnn experiment`
- `clean repo ignore rules and docs`

## Before Pushing

Before publishing the repository, check that:

1. dataset files are not staged
2. checkpoints and logs are not staged
3. generated figures are not staged
4. local IDE or virtual-environment files are not staged
5. README and docs still describe how to reproduce the project from source only
