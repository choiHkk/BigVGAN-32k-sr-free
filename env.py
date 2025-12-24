"""Environment utilities for BigVGAN configuration management.

This module provides utilities for handling hyperparameters as attribute
dictionaries and managing build environments for training.

Adapted from https://github.com/jik876/hifi-gan under the MIT license.
LICENSE is in incl_licenses directory.
"""

import os
import shutil


class AttrDict(dict):
    """Dictionary subclass that allows attribute-style access to items.

    This class extends dict to allow accessing dictionary keys as attributes,
    which is convenient for configuration and hyperparameter management.

    Example:
        >>> config = AttrDict({"learning_rate": 0.001, "batch_size": 32})
        >>> print(config.learning_rate)  # 0.001
        >>> print(config["batch_size"])  # 32
    """

    def __init__(self, *args, **kwargs):
        """Initialize AttrDict with dict-style arguments.

        Args:
            *args: Positional arguments passed to dict constructor.
            **kwargs: Keyword arguments passed to dict constructor.
        """
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config, config_name, path):
    """Build training environment by copying config file to output directory.

    Creates the output directory if it doesn't exist and copies the
    configuration file to enable reproducibility.

    Args:
        config: Source path of the configuration file.
        config_name: Name for the copied configuration file.
        path: Destination directory for the configuration file.
    """
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))
