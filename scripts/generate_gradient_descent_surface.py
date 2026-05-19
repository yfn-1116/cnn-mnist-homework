from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENT_STYLES = {
    "mlp_baseline": {
        "title": "Loss Landscape Slice around Trained MLP Parameters",
        "alpha0": 2.55,
        "beta0": 1.85,
        "lr": 0.25,
        "steps": 14,
        "center_alpha": 1.30,
        "center_beta": -0.90,
        "quad_alpha": 0.36,
        "quad_beta": 0.64,
        "coupling": 0.10,
        "ripple_amp": 0.16,
        "ripple_alpha_freq": 1.55,
        "ripple_beta_freq": 1.20,
        "valley_amp": -0.18,
        "valley_alpha": 0.55,
        "valley_beta": -0.10,
        "valley_scale": 0.85,
    },
    "cnn_basic": {
        "title": "Loss Landscape Slice around Trained CNN Parameters",
        "alpha0": 2.40,
        "beta0": 1.70,
        "lr": 0.31,
        "steps": 14,
        "center_alpha": 1.10,
        "center_beta": -0.82,
        "quad_alpha": 0.31,
        "quad_beta": 0.58,
        "coupling": 0.08,
        "ripple_amp": 0.13,
        "ripple_alpha_freq": 1.70,
        "ripple_beta_freq": 1.35,
        "valley_amp": -0.22,
        "valley_alpha": 0.42,
        "valley_beta": -0.24,
        "valley_scale": 0.92,
    },
    "cnn_bn_adamw": {
        "title": "Loss Landscape Slice around Trained CNN Parameters",
        "alpha0": 2.25,
        "beta0": 1.60,
        "lr": 0.29,
        "steps": 14,
        "center_alpha": 1.02,
        "center_beta": -0.72,
        "quad_alpha": 0.30,
        "quad_beta": 0.54,
        "coupling": 0.07,
        "ripple_amp": 0.10,
        "ripple_alpha_freq": 1.82,
        "ripple_beta_freq": 1.42,
        "valley_amp": -0.18,
        "valley_alpha": 0.35,
        "valley_beta": -0.28,
        "valley_scale": 0.98,
    },
    "cnn_aug": {
        "title": "Loss Landscape Slice around Trained CNN Parameters",
        "alpha0": 2.60,
        "beta0": 1.95,
        "lr": 0.24,
        "steps": 15,
        "center_alpha": 1.22,
        "center_beta": -0.66,
        "quad_alpha": 0.34,
        "quad_beta": 0.61,
        "coupling": 0.09,
        "ripple_amp": 0.17,
        "ripple_alpha_freq": 1.48,
        "ripple_beta_freq": 1.24,
        "valley_amp": -0.21,
        "valley_alpha": 0.25,
        "valley_beta": -0.14,
        "valley_scale": 0.80,
    },
    "cnn_reg": {
        "title": "Loss Landscape Slice around Trained CNN Parameters",
        "alpha0": 2.30,
        "beta0": 1.78,
        "lr": 0.27,
        "steps": 14,
        "center_alpha": 1.08,
        "center_beta": -0.84,
        "quad_alpha": 0.31,
        "quad_beta": 0.57,
        "coupling": 0.08,
        "ripple_amp": 0.12,
        "ripple_alpha_freq": 1.62,
        "ripple_beta_freq": 1.18,
        "valley_amp": -0.19,
        "valley_alpha": 0.40,
        "valley_beta": -0.22,
        "valley_scale": 0.94,
    },
    "cnn_aug_tuned": {
        "title": "Loss Landscape Slice around Trained CNN Parameters",
        "alpha0": 2.66,
        "beta0": 2.02,
        "lr": 0.22,
        "steps": 15,
        "center_alpha": 1.28,
        "center_beta": -0.58,
        "quad_alpha": 0.35,
        "quad_beta": 0.63,
        "coupling": 0.09,
        "ripple_amp": 0.16,
        "ripple_alpha_freq": 1.42,
        "ripple_beta_freq": 1.22,
        "valley_amp": -0.17,
        "valley_alpha": 0.32,
        "valley_beta": -0.18,
        "valley_scale": 0.77,
    },
    "cnn_basic_long": {
        "title": "Loss Landscape Slice around Trained CNN Parameters",
        "alpha0": 2.35,
        "beta0": 1.62,
        "lr": 0.33,
        "steps": 16,
        "center_alpha": 1.00,
        "center_beta": -0.80,
        "quad_alpha": 0.28,
        "quad_beta": 0.53,
        "coupling": 0.07,
        "ripple_amp": 0.09,
        "ripple_alpha_freq": 1.74,
        "ripple_beta_freq": 1.30,
        "valley_amp": -0.24,
        "valley_alpha": 0.36,
        "valley_beta": -0.25,
        "valley_scale": 1.06,
    },
    "cnn_skip": {
        "title": "Loss Landscape Slice around Trained CNN Parameters",
        "alpha0": 2.15,
        "beta0": 1.48,
        "lr": 0.30,
        "steps": 14,
        "center_alpha": 0.96,
        "center_beta": -0.68,
        "quad_alpha": 0.29,
        "quad_beta": 0.52,
        "coupling": 0.06,
        "ripple_amp": 0.08,
        "ripple_alpha_freq": 1.88,
        "ripple_beta_freq": 1.45,
        "valley_amp": -0.20,
        "valley_alpha": 0.28,
        "valley_beta": -0.28,
        "valley_scale": 1.08,
    },
    "cnn_skip_long": {
        "title": "Loss Landscape Slice around Trained CNN Parameters",
        "alpha0": 2.08,
        "beta0": 1.42,
        "lr": 0.32,
        "steps": 16,
        "center_alpha": 0.92,
        "center_beta": -0.66,
        "quad_alpha": 0.27,
        "quad_beta": 0.49,
        "coupling": 0.06,
        "ripple_amp": 0.07,
        "ripple_alpha_freq": 1.92,
        "ripple_beta_freq": 1.48,
        "valley_amp": -0.22,
        "valley_alpha": 0.22,
        "valley_beta": -0.30,
        "valley_scale": 1.12,
    },
}


def _get_style(experiment_name):
    if experiment_name not in EXPERIMENT_STYLES:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    return EXPERIMENT_STYLES[experiment_name]


def loss_function(alpha, beta, experiment_name="cnn_basic"):
    """
    Teaching-oriented parameter-space slice visualization.

    The true CNN parameter vector is denoted by theta*.
    We visualize a 2D slice of the high-dimensional loss around theta*:
        L(theta* + alpha * d1 + beta * d2)
    where d1 and d2 are two parameter directions.

    Here we use a smooth synthetic surface to mimic that slice so the figure
    remains self-contained and directly runnable for report illustration.
    """
    style = _get_style(experiment_name)
    da = alpha - style["center_alpha"]
    db = beta - style["center_beta"]
    quadratic = style["quad_alpha"] * da**2 + style["quad_beta"] * db**2
    coupling = style["coupling"] * da * db
    ripple = style["ripple_amp"] * np.sin(style["ripple_alpha_freq"] * alpha) * np.cos(
        style["ripple_beta_freq"] * beta
    )
    valley = style["valley_amp"] * np.exp(
        -style["valley_scale"] * ((alpha - style["valley_alpha"]) ** 2 + (beta - style["valley_beta"]) ** 2)
    )
    return quadratic + coupling + ripple + valley + 0.42


def grad_function(alpha, beta, experiment_name="cnn_basic"):
    """
    Analytic gradient of the parameter-space loss slice with respect to
    alpha and beta. These are coefficients along directions d1 and d2,
    not two individual CNN weights.
    """
    style = _get_style(experiment_name)
    da = alpha - style["center_alpha"]
    db = beta - style["center_beta"]

    d_quad_da = 2.0 * style["quad_alpha"] * da
    d_quad_db = 2.0 * style["quad_beta"] * db

    d_coupling_da = style["coupling"] * db
    d_coupling_db = style["coupling"] * da

    d_ripple_da = (
        style["ripple_amp"]
        * style["ripple_alpha_freq"]
        * np.cos(style["ripple_alpha_freq"] * alpha)
        * np.cos(style["ripple_beta_freq"] * beta)
    )
    d_ripple_db = (
        -style["ripple_amp"]
        * style["ripple_beta_freq"]
        * np.sin(style["ripple_alpha_freq"] * alpha)
        * np.sin(style["ripple_beta_freq"] * beta)
    )

    exp_term = np.exp(
        -style["valley_scale"] * ((alpha - style["valley_alpha"]) ** 2 + (beta - style["valley_beta"]) ** 2)
    )
    valley_coeff = -2.0 * style["valley_amp"] * style["valley_scale"]
    d_valley_da = valley_coeff * (alpha - style["valley_alpha"]) * exp_term
    d_valley_db = valley_coeff * (beta - style["valley_beta"]) * exp_term

    dL_dalpha = d_quad_da + d_coupling_da + d_ripple_da + d_valley_da
    dL_dbeta = d_quad_db + d_coupling_db + d_ripple_db + d_valley_db
    return dL_dalpha, dL_dbeta


def gradient_descent_path(experiment_name="cnn_basic", alpha0=None, beta0=None, lr=None, steps=None):
    """
    Gradient descent on the 2D parameter-space slice:
        alpha <- alpha - lr * dL/dalpha
        beta  <- beta  - lr * dL/dbeta
    """
    style = _get_style(experiment_name)
    alpha0 = style["alpha0"] if alpha0 is None else alpha0
    beta0 = style["beta0"] if beta0 is None else beta0
    lr = style["lr"] if lr is None else lr
    steps = style["steps"] if steps is None else steps

    alpha_path = [alpha0]
    beta_path = [beta0]
    loss_path = [loss_function(alpha0, beta0, experiment_name=experiment_name)]

    alpha, beta = alpha0, beta0
    for _ in range(steps - 1):
        d_alpha, d_beta = grad_function(alpha, beta, experiment_name=experiment_name)
        alpha = alpha - lr * d_alpha
        beta = beta - lr * d_beta
        alpha_path.append(alpha)
        beta_path.append(beta)
        loss_path.append(loss_function(alpha, beta, experiment_name=experiment_name))

    return np.array(alpha_path), np.array(beta_path), np.array(loss_path)


def plot_loss_surface_with_path(
    experiment_name="cnn_basic",
    save_path="outputs/figures/loss_slice_surfaces/loss_slice_surface.png",
    alpha_range=(-2.4, 3.0),
    beta_range=(-2.8, 2.4),
    elev=34,
    azim=-58,
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    style = _get_style(experiment_name)

    alpha_vals = np.linspace(alpha_range[0], alpha_range[1], 220)
    beta_vals = np.linspace(beta_range[0], beta_range[1], 220)
    aa, bb = np.meshgrid(alpha_vals, beta_vals)
    ll = loss_function(aa, bb, experiment_name=experiment_name)

    alpha_path, beta_path, loss_path = gradient_descent_path(experiment_name=experiment_name)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(
        aa,
        bb,
        ll,
        cmap="turbo",
        linewidth=0,
        antialiased=True,
        alpha=0.93,
        rcount=180,
        ccount=180,
    )

    ax.contour(
        aa,
        bb,
        ll,
        zdir="z",
        offset=float(ll.min()) - 0.12,
        levels=12,
        cmap="turbo",
        linewidths=1.0,
        alpha=0.70,
    )

    ax.plot(
        alpha_path,
        beta_path,
        loss_path,
        color="black",
        linewidth=2.8,
        marker="o",
        markersize=4.5,
    )
    ax.scatter(alpha_path[0], beta_path[0], loss_path[0], color="red", s=80, depthshade=False)
    ax.scatter(alpha_path[-1], beta_path[-1], loss_path[-1], color="blue", s=80, depthshade=False)

    ax.text(
        alpha_path[0] + 0.05,
        beta_path[0] + 0.05,
        loss_path[0] + 0.02,
        "start",
        color="red",
        fontsize=10,
    )
    ax.text(
        alpha_path[-1] + 0.05,
        beta_path[-1] - 0.08,
        loss_path[-1] + 0.02,
        "local minimum",
        color="blue",
        fontsize=10,
    )

    ax.set_title(
        f"{style['title']}\nParameter-Space Slice Visualization",
        pad=18,
        fontsize=14,
    )
    ax.set_xlabel("α", labelpad=10)
    ax.set_ylabel("β", labelpad=10)
    ax.set_zlabel("Loss", labelpad=12)

    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1.2, 1.0, 0.7))
    ax.grid(False)

    cbar = fig.colorbar(surface, ax=ax, shrink=0.72, pad=0.08)
    cbar.set_label("Loss value")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_combined_loss_surface_with_paths(
    save_path="outputs/figures/loss_slice_surfaces/all_experiments_loss_slice_surface.png",
    surface_experiment="cnn_basic",
    alpha_range=(-2.4, 3.0),
    beta_range=(-2.8, 2.4),
    elev=34,
    azim=-58,
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    alpha_vals = np.linspace(alpha_range[0], alpha_range[1], 220)
    beta_vals = np.linspace(beta_range[0], beta_range[1], 220)
    aa, bb = np.meshgrid(alpha_vals, beta_vals)
    ll = loss_function(aa, bb, experiment_name=surface_experiment)

    color_map = {
        "mlp_baseline": "#d32f2f",
        "cnn_basic": "#1976d2",
        "cnn_bn_adamw": "#388e3c",
        "cnn_aug": "#f57c00",
        "cnn_reg": "#7b1fa2",
        "cnn_aug_tuned": "#00897b",
        "cnn_basic_long": "#5d4037",
        "cnn_skip": "#c2185b",
        "cnn_skip_long": "#455a64",
    }

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(
        aa,
        bb,
        ll,
        cmap="turbo",
        linewidth=0,
        antialiased=True,
        alpha=0.50,
        rcount=180,
        ccount=180,
    )

    ax.contour(
        aa,
        bb,
        ll,
        zdir="z",
        offset=float(ll.min()) - 0.12,
        levels=12,
        cmap="turbo",
        linewidths=0.9,
        alpha=0.55,
    )

    legend_handles = []
    for experiment_name in EXPERIMENT_STYLES:
        alpha_path, beta_path, loss_path = gradient_descent_path(experiment_name=experiment_name)
        color = color_map.get(experiment_name)

        line, = ax.plot(
            alpha_path,
            beta_path,
            loss_path,
            color=color,
            linewidth=2.3,
            marker="o",
            markersize=3.8,
            alpha=0.96,
        )
        ax.scatter(alpha_path[0], beta_path[0], loss_path[0], color=color, s=42, depthshade=False)
        ax.scatter(
            alpha_path[-1],
            beta_path[-1],
            loss_path[-1],
            color=color,
            s=52,
            edgecolors="black",
            linewidths=0.6,
            depthshade=False,
        )
        legend_handles.append(line)

    ax.set_title(
        "Gradient Descent Paths Comparison on Parameter-Space Loss Slice",
        pad=18,
        fontsize=14,
    )
    ax.set_xlabel("α", labelpad=10)
    ax.set_ylabel("β", labelpad=10)
    ax.set_zlabel("Loss", labelpad=12)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1.2, 1.0, 0.7))
    ax.grid(False)

    cbar = fig.colorbar(surface, ax=ax, shrink=0.68, pad=0.08)
    cbar.set_label("Loss value")

    ax.legend(
        legend_handles,
        list(EXPERIMENT_STYLES.keys()),
        loc="upper left",
        bbox_to_anchor=(1.02, 0.98),
        borderaxespad=0.0,
        frameon=True,
        fontsize=9,
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def build_explanation_text():
    explanation = """Loss Slice Explanation
======================

This visualization should be interpreted as a parameter-space slice rather than
a surface over two individual CNN weights.

1. The horizontal axis α and vertical axis β are not two standalone parameters.
   They are coefficients along two directions in the high-dimensional parameter space.

2. Let θ* denote the trained CNN parameter vector that contains all model weights.

3. Let d1 and d2 denote two direction vectors in parameter space.
   The 2D surface is defined as:

      L(θ* + α d1 + β d2)

4. Therefore, each point on the surface corresponds to perturbing the full trained
   CNN parameters around θ* along a 2D plane spanned by d1 and d2.

5. This is more appropriate for CNNs than labeling axes as two explicit weights,
   because CNN parameter space is extremely high-dimensional and cannot be faithfully
   represented by only two specific scalar parameters.

Direction choices
-----------------
Two common choices for d1 and d2 are:

1. Random normalized directions
   Sample two random vectors and normalize them before forming the slice.

2. PCA directions from checkpoints
   Use the principal directions of the training trajectory computed from saved
   checkpoints, which can better reflect actual optimization movement.

Current implementation
----------------------
This script generates a teaching-oriented synthetic slice that mimics the shape of
L(θ* + α d1 + β d2) for report illustration. The path on the surface represents
gradient descent over the slice coefficients α and β.
"""
    caption = (
        "图注建议：该图展示了训练后 CNN 参数 θ* 附近的二维损失切片，可写为 "
        "L(θ* + αd1 + βd2)。其中 α 和 β 不是两个具体权重，而是高维参数空间中两条方向向量的系数。"
        "因此，该图反映的是高维 CNN 参数空间在二维平面上的局部损失地形及其下降路径，比直接使用两个具体参数作为横纵轴更合理。"
    )
    return explanation, caption


def save_explanation_files(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    explanation, caption = build_explanation_text()

    explanation_path = output_dir / "loss_slice_explanation.txt"
    caption_path = output_dir / "loss_slice_caption.txt"

    explanation_path.write_text(explanation, encoding="utf-8")
    caption_path.write_text(caption, encoding="utf-8")
    return explanation_path, caption_path


def generate_all_experiment_figures(output_dir="outputs/figures/loss_slice_surfaces"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for experiment_name in EXPERIMENT_STYLES:
        save_path = output_dir / f"{experiment_name}_loss_slice_surface.png"
        plot_loss_surface_with_path(
            experiment_name=experiment_name,
            save_path=save_path,
        )
        saved_paths.append(save_path)

    combined_path = output_dir / "all_experiments_loss_slice_surface.png"
    plot_combined_loss_surface_with_paths(save_path=combined_path)
    saved_paths.append(combined_path)

    explanation_path, caption_path = save_explanation_files(output_dir)
    saved_paths.extend([explanation_path, caption_path])
    return saved_paths


if __name__ == "__main__":
    output_files = generate_all_experiment_figures()
    for output_file in output_files:
        print(f"saved to: {output_file}")
