"""
可视化工具函数
"""

import os
from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

TITLE_FONT_SIZE = 16


def plot_samples(
        samples1: np.ndarray,
        samples1_name: str,
        samples2: Optional[np.ndarray] = None,
        samples2_name: Optional[str] = None,
        num_samples: int = 5,
) -> None:
    """
    Plot one or two sets of samples.

    Args:
        samples1 (np.ndarray): The first set of samples to plot.
        samples1_name (str): The name for the first set of samples in the plot title.
        samples2 (Optional[np.ndarray]): The second set of samples to plot.
                                         Defaults to None.
        samples2_name (Optional[str]): The name for the second set of samples in the
                                       plot title.
                                       Defaults to None.
        num_samples (int, optional): The number of samples to plot.
                                     Defaults to 5.

    Returns:
        None
    """
    if samples2 is not None:
        fig, axs = plt.subplots(num_samples, 2, figsize=(10, 6))
    else:
        fig, axs = plt.subplots(num_samples, 1, figsize=(6, 8))

    for i in range(num_samples):
        rnd_idx1 = np.random.choice(len(samples1))
        sample1 = samples1[rnd_idx1]

        if samples2 is not None:
            rnd_idx2 = np.random.choice(len(samples2))
            sample2 = samples2[rnd_idx2]

            axs[i, 0].plot(sample1)
            axs[i, 0].set_title(samples1_name)

            axs[i, 1].plot(sample2)
            axs[i, 1].set_title(samples2_name)
        else:
            axs[i].plot(sample1)
            axs[i].set_title(samples1_name)

    if samples2 is not None:
        fig.suptitle(f"{samples1_name} vs {samples2_name}", fontsize=TITLE_FONT_SIZE)
    else:
        fig.suptitle(samples1_name, fontsize=TITLE_FONT_SIZE)

    fig.tight_layout()
    plt.show()


def plot_latent_space_samples(vae, n: int, figsize: tuple) -> None:
    """
    Plot samples from a 2D latent space.

    Args:
        vae: The VAE models with a method to generate samples from latent space.
        n (int): Number of points in each dimension of the grid.
        figsize (tuple): Figure size for the plot.
    """
    scale = 3.0
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]
    grid_size = len(grid_x)

    # Generate the latent space grid
    Z2 = np.array([[x, y] for x in grid_x for y in grid_y])

    # Generate samples from the VAE given the latent space coordinates
    X_recon = vae.get_prior_samples_given_Z(Z2)
    X_recon = np.squeeze(X_recon)

    fig, axs = plt.subplots(grid_size, grid_size, figsize=figsize)

    # Plot each generated sample
    for k, (i, yi) in enumerate(enumerate(grid_y)):
        for j, xi in enumerate(grid_x):
            axs[i, j].plot(X_recon[k])
            axs[i, j].set_title(f"z1={np.round(xi, 2)}; z2={np.round(yi, 2)}")
            k += 1

    fig.suptitle("Generated Samples From 2D Embedded Space", fontsize=TITLE_FONT_SIZE)
    fig.tight_layout()
    # plt.show()


def avg_over_dim(data: np.ndarray, axis: int) -> np.ndarray:
    """
    Average over the feature dimension of the data.

    Args:
        data (np.ndarray): The data to average over.
        axis (int): Axis to average over.

    Returns:
        np.ndarray: The data averaged over the feature dimension.
    """
    return np.mean(data, axis=axis)


def visualize_and_save_tsne(
        samples1: np.ndarray,
        samples1_name: str,
        samples2: np.ndarray,
        samples2_name: str,
        scenario_name: str,
        save_dir: str,
        max_samples: int = 1000,
        perplexity: Optional[int] = None,
) -> None:
    """
    Visualize the t-SNE of two sets of samples and save to file.
    根据样本数自动调整perplexity参数。

    Args:
        samples1 (np.ndarray): The first set of samples to plot.
        samples1_name (str): The name for the first set of samples in the plot title.
        samples2 (np.ndarray): The second set of samples to plot.
        samples2_name (str): The name for the second set of samples in the
                             plot title.
        scenario_name (str): The scenario name for the given samples.
        save_dir (str): Dir path to which to save the file.
        max_samples (int): Maximum number of samples to use in the plot.
        perplexity (Optional[int]): t-SNE perplexity parameter. None则自动调整。
    """
    if samples1.shape != samples2.shape:
        raise ValueError(
            "Given pairs of samples dont match in shapes. Cannot create t-SNE.\n"
            f"sample1 shape: {samples1.shape}; sample2 shape: {samples2.shape}"
        )

    samples1_2d = avg_over_dim(samples1, axis=2)
    samples2_2d = avg_over_dim(samples2, axis=2)

    # num of samples used in the t-SNE plot
    n_samples1 = min(samples1_2d.shape[0], max_samples)
    n_samples2 = min(samples2_2d.shape[0], max_samples)

    # 如果样本数量太少，跳过t-SNE
    if n_samples1 < 5 or n_samples2 < 5:
        print(f"[WARNING] Skipping t-SNE: insufficient samples "
              f"(samples1: {n_samples1}, samples2: {n_samples2})")
        return

    # 随机选择样本
    idx1 = np.random.choice(samples1_2d.shape[0], n_samples1, replace=False)
    idx2 = np.random.choice(samples2_2d.shape[0], n_samples2, replace=False)

    selected_samples1 = samples1_2d[idx1]
    selected_samples2 = samples2_2d[idx2]

    # Combine the original and generated samples
    combined_samples = np.vstack([selected_samples1, selected_samples2])
    total_samples = combined_samples.shape[0]

    # 动态调整perplexity
    if perplexity is None:
        perplexity = min(30, max(5, total_samples - 1))
    elif perplexity >= total_samples:
        perplexity = max(5, total_samples - 1)
        print(f"[WARNING] Adjusted perplexity to {perplexity} due to small sample size ({total_samples})")

    print(f"[INFO] t-SNE with {total_samples} samples, perplexity={perplexity}")

    try:
        # Compute the t-SNE of the combined samples
        tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=300, random_state=42)
        tsne_samples = tsne.fit_transform(combined_samples)

        # Plot the t-SNE samples
        plt.figure(figsize=(10, 8))

        # 绘制原始样本
        plt.scatter(
            tsne_samples[:n_samples1, 0], tsne_samples[:n_samples1, 1],
            c='blue', alpha=0.6, s=50, label=samples1_name, edgecolors='black', linewidth=0.5
        )

        # 绘制生成样本
        plt.scatter(
            tsne_samples[n_samples1:, 0], tsne_samples[n_samples1:, 1],
            c='red', alpha=0.6, s=50, label=samples2_name, marker='^', edgecolors='black', linewidth=0.5
        )

        plt.title(f"t-SNE Visualization: {scenario_name}", fontsize=TITLE_FONT_SIZE)
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save the plot to a file
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{scenario_name}_tsne.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"[INFO] t-SNE plot saved to: {save_path}")

    except Exception as e:
        print(f"[ERROR] Failed to perform t-SNE: {str(e)}")

        # 如果t-SNE失败，尝试使用PCA
        print("[INFO] Trying PCA as an alternative...")
        try:
            pca = PCA(n_components=2, random_state=42)
            pca_samples = pca.fit_transform(combined_samples)

            plt.figure(figsize=(10, 8))

            plt.scatter(
                pca_samples[:n_samples1, 0], pca_samples[:n_samples1, 1],
                c='blue', alpha=0.6, s=50, label=samples1_name, edgecolors='black', linewidth=0.5
            )

            plt.scatter(
                pca_samples[n_samples1:, 0], pca_samples[n_samples1:, 1],
                c='red', alpha=0.6, s=50, label=samples2_name, marker='^', edgecolors='black', linewidth=0.5
            )

            plt.title(f"PCA Visualization: {scenario_name}", fontsize=TITLE_FONT_SIZE)
            plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
            plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
            plt.legend()
            plt.grid(True, alpha=0.3)

            save_path = os.path.join(save_dir, f"{scenario_name}_pca.png")
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"[INFO] PCA plot saved to: {save_path}")

        except Exception as e2:
            print(f"[ERROR] Failed to perform PCA: {str(e2)}")