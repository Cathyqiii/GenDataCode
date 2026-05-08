import random
from typing import Union, List, Optional

import torch
import numpy as np

from vae.vae_dense_model import VariationalAutoencoderDense as VAE_Dense
from vae.vae_conv_model import VariationalAutoencoderConv as VAE_Conv
from vae.timevae import TimeVAE


def set_seeds(seed: int = 111) -> None:
    """
    Set seeds for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    # Set the seed for PyTorch
    torch.manual_seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for Python built-in random module
    random.seed(seed)


def instantiate_vae_model(
    vae_type: str, sequence_length: int, feature_dim: int, batch_size: int, **kwargs
) -> Union[VAE_Dense, VAE_Conv, TimeVAE]:
    """
    Instantiate a Variational Autoencoder (VAE) models based on the specified type.

    Args:
        vae_type (str): The type of VAE models to instantiate.
                        One of ('vae_dense', 'vae_conv', 'timeVAE').
        sequence_length (int): The sequence length.
        feature_dim (int): The feature dimension.
        batch_size (int): Batch size for training.

    Returns:
        Union[VAE_Dense, VAE_Conv, TimeVAE]: The instantiated VAE models.

    Raises:
        ValueError: If an unrecognized VAE type is provided.
    """
    set_seeds(seed=123)

    if vae_type == "vae_dense":
        vae = VAE_Dense(
            seq_len=sequence_length,
            feat_dim=feature_dim,
            batch_size=batch_size,
            **kwargs,
        )
    elif vae_type == "vae_conv":
        vae = VAE_Conv(
            seq_len=sequence_length,
            feat_dim=feature_dim,
            batch_size=batch_size,
            **kwargs,
        )
    elif vae_type == "timeVAE":
        vae = TimeVAE(
            seq_len=sequence_length,
            feat_dim=feature_dim,
            batch_size=batch_size,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unrecognized models type [{vae_type}]. "
            "Please choose from vae_dense, vae_conv, timeVAE."
        )

    return vae


def train_vae(vae, train_data, max_epochs, verbose=0):
    """
    Train a VAE models.

    Args:
        vae (Union[VAE_Dense, VAE_Conv, TimeVAE]): The VAE models to train.
        train_data (np.ndarray): The training data which must be of shape
                                 [num_samples, window_len, feature_dim].
        max_epochs (int, optional): The maximum number of epochs to train
                                    the models.
                                    Defaults to 100.
        verbose (int, optional): Verbose arg for keras models.fit()
    """
    vae.fit_on_data(train_data, max_epochs, verbose)


def save_vae_model(vae, dir_path: str) -> None:
    """
    Save the weights of a VAE models.

    Args:
        vae (Union[VAE_Dense, VAE_Conv, TimeVAE]): The VAE models to save.
        dir_path (str): The directory to save the models weights.
    """
    vae.save(dir_path)


def load_vae_model(vae_type: str, dir_path: str) -> Union[VAE_Dense, VAE_Conv, TimeVAE]:
    """
    Load a VAE models from the specified directory.

    Args:
        vae_type (str): The type of VAE models to load.
                        One of ('vae_dense', 'vae_conv', 'timeVAE').
        dir_path (str): The directory containing the models weights.

    Returns:
        Union[VAE_Dense, VAE_Conv, TimeVAE]: The loaded VAE models.
    """
    if vae_type == "vae_dense":
        vae = VAE_Dense.load(dir_path)
    elif vae_type == "vae_conv":
        vae = VAE_Conv.load(dir_path)
    elif vae_type == "timeVAE":
        vae = TimeVAE.load(dir_path)
    else:
        raise ValueError(
            f"Unrecognized models type [{vae_type}]. "
            "Please choose from vae_dense, vae_conv, timeVAE."
        )

    return vae


def get_posterior_samples(vae, data):
    """
    Get posterior samples from the VAE models.

    Args:
        vae (Union[VAE_Dense, VAE_Conv, TimeVAE]): The trained VAE models.
        data (np.ndarray): The data to generate posterior samples from.

    Returns:
        np.ndarray: The posterior samples.
    """
    return vae.predict(data)


def get_prior_samples(vae, num_samples: int):
    """
    Get prior samples from the VAE models.

    Args:
        vae (Union[VAE_Dense, VAE_Conv, TimeVAE]): The trained VAE models.
        num_samples (int): The number of samples to generate.

    Returns:
        np.ndarray: The prior samples.
    """
    return vae.get_prior_samples(num_samples=num_samples)
