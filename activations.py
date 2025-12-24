"""Sine-based periodic activation functions for neural networks.

This module provides Snake and SnakeBeta activation functions, which use
sine-based periodic components with learnable frequency parameters. These
activations are particularly effective for audio synthesis tasks.

Implementation adapted from https://github.com/EdwardDixon/snake under the MIT license.
LICENSE is in incl_licenses directory.

References:
    Liu Ziyin, Tilman Hartwig, Masahito Ueda. "Neural Networks Fail to Learn
    Periodic Functions and How to Fix It." https://arxiv.org/abs/2006.08195
"""

import torch
from torch import nn, pow, sin
from torch.nn import Parameter


class Snake(nn.Module):
    """Sine-based periodic activation function with trainable frequency.

    Applies the Snake activation: x + (1/alpha) * sin^2(alpha * x)

    The alpha parameter controls the frequency of the periodic component
    and is learned during training.

    Args:
        in_features: Number of input features (channels).
        alpha: Initial value for the frequency parameter. Defaults to 1.0.
        alpha_trainable: Whether alpha should be trainable. Defaults to True.
        alpha_logscale: If True, alpha is stored in log scale. Defaults to False.

    Attributes:
        alpha: Trainable frequency parameter of shape (in_features,).
        alpha_logscale: Whether alpha is stored in log scale.

    Example:
        >>> activation = Snake(256)
        >>> x = torch.randn(32, 256, 100)  # (B, C, T)
        >>> output = activation(x)  # Same shape as input
    """

    def __init__(
        self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False
    ):
        """Initialize Snake activation.

        Args:
            in_features: Number of input features (channels).
            alpha: Initial value for the frequency parameter. Defaults to 1.0.
                Higher values result in higher-frequency periodic components.
            alpha_trainable: Whether alpha should be trainable. Defaults to True.
            alpha_logscale: If True, alpha is stored in log scale for numerical
                stability. Defaults to False.
        """
        super(Snake, self).__init__()
        self.in_features = in_features

        # Initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # Log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else:  # Linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """Apply Snake activation to input tensor.

        Computes: x + (1/alpha) * sin^2(alpha * x)

        Args:
            x: Input tensor of shape (B, C, T).

        Returns:
            torch.Tensor: Activated tensor of the same shape as input.
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # Line up with x to [B, C, T]
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        x = x + (1.0 / (alpha + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x


class SnakeBeta(nn.Module):
    """Modified Snake activation with separate frequency and magnitude parameters.

    Applies the SnakeBeta activation: x + (1/beta) * sin^2(alpha * x)

    Unlike Snake, this variant uses separate trainable parameters for
    frequency (alpha) and magnitude (beta) of the periodic component.

    Args:
        in_features: Number of input features (channels).
        alpha: Initial value for both frequency and magnitude parameters.
            Defaults to 1.0.
        alpha_trainable: Whether parameters should be trainable. Defaults to True.
        alpha_logscale: If True, parameters are stored in log scale. Defaults to False.

    Attributes:
        alpha: Trainable frequency parameter of shape (in_features,).
        beta: Trainable magnitude parameter of shape (in_features,).
        alpha_logscale: Whether parameters are stored in log scale.

    Example:
        >>> activation = SnakeBeta(256)
        >>> x = torch.randn(32, 256, 100)  # (B, C, T)
        >>> output = activation(x)  # Same shape as input
    """

    def __init__(
        self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False
    ):
        """Initialize SnakeBeta activation.

        Args:
            in_features: Number of input features (channels).
            alpha: Initial value for both frequency and magnitude parameters.
                Defaults to 1.0. Higher values result in higher-frequency and
                higher-magnitude periodic components.
            alpha_trainable: Whether parameters should be trainable. Defaults to True.
            alpha_logscale: If True, parameters are stored in log scale for
                numerical stability. Defaults to False.
        """
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # Initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # Log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:  # Linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """Apply SnakeBeta activation to input tensor.

        Computes: x + (1/beta) * sin^2(alpha * x)

        Args:
            x: Input tensor of shape (B, C, T).

        Returns:
            torch.Tensor: Activated tensor of the same shape as input.
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # Line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x
