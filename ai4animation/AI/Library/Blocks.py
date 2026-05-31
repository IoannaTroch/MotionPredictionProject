# Copyright (c) Meta Platforms, Inc. and affiliates.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ai4animation.AI import Manifolds
from ai4animation.AI.Library import Defaults
from ai4animation.AI.Library.Layers import (
    FiLMLayer,
    FiLMLinearLayer,
    LinearLayer,
    VariationalLayer,
)
from einops import rearrange
from torch.nn.parameter import Parameter


class LinearBlock(torch.nn.Module):
    """NN with 3 linear layers"""
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        dropout=Defaults.Dropout,
        activation=Defaults.Activation,
    ):
        super(LinearBlock, self).__init__()

        self.L1 = LinearLayer(input_size, hidden_size, dropout, activation)
        self.L2 = LinearLayer(hidden_size, hidden_size, dropout, activation)
        self.L3 = LinearLayer(hidden_size, output_size, dropout, None)

    def forward(self, z):
        z = self.L1(z)
        z = self.L2(z)
        z = self.L3(z)
        return z


class FiLMLinearBlock(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        film_size,
        dropout=Defaults.Dropout,
        activation=Defaults.Activation,
    ):
        super(FiLMLinearBlock, self).__init__()

        self.L1 = FiLMLinearLayer(
            input_size, hidden_size, film_size, dropout, activation
        )
        self.L2 = FiLMLinearLayer(
            hidden_size, hidden_size, film_size, dropout, activation
        )
        self.L3 = FiLMLinearLayer(hidden_size, output_size, film_size, dropout, None)

    def forward(self, z, film):
        z = self.L1(z, film)
        z = self.L2(z, film)
        z = self.L3(z, film)
        return z


class RegularizedFiLMLinearBlock(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        regularization_size,
        film_size,
        dropout=Defaults.Dropout,
        activation=Defaults.Activation,
    ):
        super(RegularizedFiLMLinearBlock, self).__init__()

        self.L1 = FiLMLinearLayer(
            input_size, hidden_size, film_size, dropout, activation
        )
        self.L2 = FiLMLinearLayer(
            hidden_size, hidden_size, film_size, dropout, activation
        )
        self.L3 = FiLMLinearLayer(hidden_size, output_size, film_size, dropout, None)
        self.R = FiLMLinearLayer(
            hidden_size, regularization_size, film_size, dropout, None
        )

    def forward(self, z, film):
        z = self.L1(z, film)
        z = self.L2(z, film)
        y = self.L3(z, film)
        if self.training:
            r = self.R(z, film)
            return y, r
        else:
            return y
