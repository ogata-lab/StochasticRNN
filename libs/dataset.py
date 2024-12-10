#
# Copyright (c) Since 2024 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(seed=47)


def generate_waveforms(P=25, total_steps=1000):
    """
    Generates base waveforms y1, y2, y3, y4 based on periodic functions.

    Args:
        P (int): Period of the waveforms.
        total_steps (int): Total number of time steps.

    Returns:
        tuple: Four numpy arrays representing waveforms y1, y2, y3, y4.
    """
    t = np.linspace(0, total_steps, total_steps)
    y1 = -0.4 * (np.cos(2 * np.pi * t / P) - 1)
    y2 = 0.8 * np.sin(2 * np.pi * t / P)
    y3 = -0.4 * (np.cos(4 * np.pi * t / P) - 1)
    y4 = 0.8 * np.sin(4 * np.pi * t / P)
    return y1, y2, y3, y4


def get_dataset():
    """
    Creates a dataset by adding noise to combinations of waveforms.

    Returns:
        tuple:
            dataset (np.array): A dataset of noisy waveform pairs.
            noise_levels (list): A list of noise levels corresponding to the waveform pairs.
    """
    y1, y2, y3, y4 = generate_waveforms()
    noise_levels = [0.01, 0.03, 0.05, 0.07]

    # Define waveform combinations
    waveform_pairs = [
        (y1, y2),
        (-y1, y2),
        (y2, y1),
        (y2, -y1),
        (y1, y4),
        (-y1, y4),
        (y4, y1),
        (y4, -y1),
        (y3, y2),
        (-y3, y2),
        (y2, y3),
        (y2, -y3),
    ]

    # Extend noise levels to match the number of waveform pairs
    noise_list = noise_levels * (len(waveform_pairs) // len(noise_levels))

    # Generate noisy dataset
    dataset = []
    for (x1, x2), noise in zip(waveform_pairs, noise_list):
        x1_noisy = x1 + np.random.normal(loc=0, scale=noise, size=x1.shape)
        x2_noisy = x2 + np.random.normal(loc=0, scale=noise, size=x2.shape)
        dataset.append(np.column_stack([x1_noisy, x2_noisy]))

    return np.array(dataset), noise_list


def plot_dataset(dataset, noise_levels):
    """
    Plots the dataset of noisy waveform pairs with noise level annotations.

    Args:
        dataset (list): A list of numpy arrays containing noisy waveform pairs.
        noise_levels (list): A list of noise levels corresponding to the dataset.
    """
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    axes = axes.flatten()

    for idx, (data, noise_level) in enumerate(zip(dataset, noise_levels)):
        axes[idx].plot(data[:, 0], data[:, 1], linewidth=0.5)
        axes[idx].set_xlim(-1, 1)
        axes[idx].set_ylim(-1, 1)
        axes[idx].set_title(f"Noise: {noise_level:.2f}", fontsize=8)
        axes[idx].tick_params(labelsize=6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dataset, noise_levels = get_dataset()
    plot_dataset(dataset, noise_levels)
