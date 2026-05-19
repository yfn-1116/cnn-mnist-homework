import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_training_curves(history: dict, save_dir: str, prefix: str):
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_loss.png"))
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_acc.png"))
    plt.close()

    if history.get("weight_l2"):
        tracked_names = history.get("tracked_weight_names", {})

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history["weight_l2"], label="Total Weight L2", linewidth=2.0)
        plt.plot(
            epochs,
            history["first_weight_l2"],
            label=f"First Layer L2 ({tracked_names.get('first', 'first weight')})",
        )
        plt.plot(
            epochs,
            history["last_weight_l2"],
            label=f"Last Layer L2 ({tracked_names.get('last', 'last weight')})",
        )
        plt.xlabel("Epoch")
        plt.ylabel("L2 Norm")
        plt.title("Weight Dynamics During Training")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}_weight_dynamics.png"))
        plt.close()


def plot_first_conv_layer_weights(model, save_dir: str, prefix: str, max_filters: int = 64):
    os.makedirs(save_dir, exist_ok=True)

    first_layer = None
    first_layer_name = None

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            first_layer = module
            first_layer_name = name
            break

    if first_layer is None or not hasattr(first_layer, "weight"):
        return None

    weights = first_layer.weight.detach().cpu().numpy()
    save_path = os.path.join(save_dir, f"{prefix}_first_conv_weights.png")
    num_filters = min(weights.shape[0], max_filters)
    cols = min(8, num_filters)
    rows = int(np.ceil(num_filters / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.8))
    axes = np.atleast_1d(axes).ravel()
    for idx, ax in enumerate(axes):
        ax.axis("off")
        if idx >= num_filters:
            continue
        kernel = weights[idx]
        kernel_img = kernel[0] if kernel.shape[0] == 1 else kernel.mean(axis=0)
        ax.imshow(kernel_img)
        ax.set_title(f"Filter {idx}", fontsize=8)

    fig.suptitle(f"First Conv Layer Weights ({first_layer_name})", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_optimization_path_3d(history: dict, save_dir: str, prefix: str, title: str | None = None):
    os.makedirs(save_dir, exist_ok=True)

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    if not train_loss or not val_loss:
        raise ValueError("history must contain non-empty train_loss and val_loss")

    epochs = np.arange(1, len(train_loss) + 1)
    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)

    fig = plt.figure(figsize=(7.2, 5.4))
    ax = fig.add_subplot(111, projection="3d")

    points = ax.scatter(
        epochs,
        train_loss,
        val_loss,
        c=epochs,
        cmap="viridis",
        s=42,
        depthshade=False,
    )
    ax.plot(epochs, train_loss, val_loss, color="#d32f2f", linewidth=2.2)

    ax.scatter(epochs[0], train_loss[0], val_loss[0], c="#1565c0", s=68, depthshade=False)
    ax.scatter(epochs[-1], train_loss[-1], val_loss[-1], c="#2e7d32", s=68, depthshade=False)

    for epoch, tr_loss, va_loss in zip(epochs, train_loss, val_loss):
        ax.text(epoch, tr_loss, va_loss, str(int(epoch)), fontsize=7)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss")
    ax.set_zlabel("Val Loss")
    ax.set_title(title or "3D Optimization Path")
    ax.view_init(elev=28, azim=-60)
    fig.colorbar(points, ax=ax, shrink=0.78, pad=0.08, label="Epoch")
    fig.tight_layout()
    save_path = os.path.join(save_dir, f"{prefix}_optimization_3d.png")
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return save_path


def _build_skip_connection_surfaces():
    x = np.linspace(-2.8, 2.8, 240)
    y = np.linspace(-2.2, 2.2, 220)
    xx, yy = np.meshgrid(x, y)

    no_skip = 0.18 * xx**4 + 0.8 * yy**2 + 0.35 * np.sin(2.2 * xx) + 0.12 * xx * yy
    with_skip = 0.22 * xx**2 + 0.28 * yy**2 + 0.04 * np.sin(1.6 * xx) + 0.03 * np.cos(1.4 * yy)

    return xx, yy, no_skip, with_skip


def generate_skip_connection_figure(save_dir: str, filename: str = "skip_connection_landscape.png"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    xx, yy, no_skip, with_skip = _build_skip_connection_surfaces()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    cmap = "coolwarm"

    panels = [
        (
            axes[0],
            no_skip,
            "Without Skip Connection",
            [(-2.1, 1.5), (-1.4, 0.6), (-0.6, 1.1), (0.2, 0.0), (0.8, -0.7)],
        ),
        (
            axes[1],
            with_skip,
            "With Skip Connection",
            [(-2.0, 1.1), (-1.2, 0.6), (-0.5, 0.25), (0.2, -0.05), (0.85, -0.3)],
        ),
    ]

    for ax, surface, title, points in panels:
        contour = ax.contourf(xx, yy, surface, levels=28, cmap=cmap)
        ax.contour(xx, yy, surface, levels=12, colors="white", linewidths=0.35, alpha=0.45)

        px = [p[0] for p in points]
        py = [p[1] for p in points]
        ax.plot(px, py, color="black", linewidth=2.0)

        for start, end in zip(points[:-1], points[1:]):
            ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.8},
            )

        ax.scatter(px[:-1], py[:-1], s=90, c=["#ff7043", "#ffca28", "#42a5f5", "#9ccc65"], zorder=3)
        ax.scatter(px[-1], py[-1], s=120, c="#5c6bc0", zorder=3)
        ax.set_title(title)
        ax.set_xlabel("Parameter Direction 1")
        ax.set_ylabel("Parameter Direction 2")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(contour, ax=axes.ravel().tolist(), shrink=0.82, label="Loss")
    fig.suptitle("Illustration of Loss Landscape and Optimization Path", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return save_path


def generate_skip_connection_figure_3d(save_dir: str, filename: str = "skip_connection_landscape_3d.png"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    xx, yy, no_skip, with_skip = _build_skip_connection_surfaces()
    cmap = "coolwarm"

    def no_skip_fn(xv, yv):
        return 0.18 * xv**4 + 0.8 * yv**2 + 0.35 * np.sin(2.2 * xv) + 0.12 * xv * yv

    def with_skip_fn(xv, yv):
        return 0.22 * xv**2 + 0.28 * yv**2 + 0.04 * np.sin(1.6 * xv) + 0.03 * np.cos(1.4 * yv)

    fig = plt.figure(figsize=(12, 5.3))
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]

    panels = [
        (
            axes[0],
            no_skip,
            no_skip_fn,
            "Without Skip Connection",
            [(-2.1, 1.5), (-1.4, 0.6), (-0.6, 1.1), (0.2, 0.0), (0.8, -0.7)],
        ),
        (
            axes[1],
            with_skip,
            with_skip_fn,
            "With Skip Connection",
            [(-2.0, 1.1), (-1.2, 0.6), (-0.5, 0.25), (0.2, -0.05), (0.85, -0.3)],
        ),
    ]

    for ax, surface, surface_fn, title, points in panels:
        ax.plot_surface(xx, yy, surface, cmap=cmap, linewidth=0, antialiased=True, alpha=0.92)
        ax.contour(xx, yy, surface, zdir="z", offset=surface.min() - 0.2, cmap=cmap, levels=12, alpha=0.8)

        px = np.array([p[0] for p in points])
        py = np.array([p[1] for p in points])
        pz = surface_fn(px, py)

        ax.plot(px, py, pz, color="black", linewidth=2.2)
        ax.scatter(px[:-1], py[:-1], pz[:-1], s=50, c=["#ff7043", "#ffca28", "#42a5f5", "#9ccc65"], depthshade=False)
        ax.scatter(px[-1], py[-1], pz[-1], s=70, c="#5c6bc0", depthshade=False)

        ax.set_title(title)
        ax.set_xlabel("Dir 1")
        ax.set_ylabel("Dir 2")
        ax.set_zlabel("Loss")
        ax.view_init(elev=31, azim=-61)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    fig.suptitle("3D Illustration of Loss Landscape and Optimization Path", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return save_path
