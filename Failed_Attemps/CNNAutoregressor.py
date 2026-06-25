# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# CNN + Autoregressive MLP for Human Motion Prediction
#
# Architecture:
#   1) CNN feature extractor — 1D conv over the time axis (WindowSize frames)
#      extracts local temporal patterns (acceleration, rhythm, curvature)
#      per joint feature dimension
#   2) Autoregressive MLP — takes CNN features and predicts the next frame
#
# Data flow:
#   [Batch, WindowSize, FrameDim]
#   → permute → [Batch, FrameDim, WindowSize]    (channels=features, length=time)
#   → Conv1D layers → [Batch, cnn_channels, WindowSize]
#   → flatten → [Batch, cnn_channels * WindowSize]
#   → MLP → [Batch, FrameDim]

import torch
import torch.nn as nn
import torch.nn.functional as F
from ai4animation.AI import Losses, Modules, Stats
from ai4animation.AI.Library import Defaults


class Model(nn.Module):
    def __init__(
        self,
        frame_dim,
        window_size,
        hidden_dim=512,
        cnn_channels=128,
        dropout=Defaults.Dropout,
    ):
        super().__init__()

        self.FrameDim    = frame_dim
        self.WindowSize  = window_size
        self.OutputDim   = frame_dim

        # Running stats on flat frames — same as AutoregressionMLP
        self.InputStats  = Stats.RunningStats(frame_dim)
        self.OutputStats = Stats.RunningStats(frame_dim)

        # ------------------------------------------------------------------
        # CNN block — extracts local temporal patterns across the window
        # Input:  [Batch, FrameDim, WindowSize]  (treat features as channels)
        # kernel_size=3, padding=1 keeps WindowSize intact
        # ------------------------------------------------------------------
        self.CNN = nn.Sequential(
            nn.Conv1d(
                in_channels=frame_dim,
                out_channels=cnn_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                in_channels=cnn_channels,
                out_channels=cnn_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # After CNN: [Batch, cnn_channels, WindowSize] → flatten
        cnn_out_dim = cnn_channels * window_size

        # ------------------------------------------------------------------
        # Autoregressive MLP — same LinearEncoder as AutoregressionMLP
        # Takes flattened CNN features, predicts one next frame
        # ------------------------------------------------------------------
        self.MLP = Modules.LinearEncoder(
            cnn_out_dim, hidden_dim, frame_dim, dropout
        )

    def _extract_features(self, x_windows_norm):
        """
        CNN feature extraction over window frames.
        x_windows_norm: [Batch, WindowSize, FrameDim] — normalised
        returns:        [Batch, cnn_out_dim]
        """
        # [B, W, FrameDim] → [B, FrameDim, W]  (channels=features, length=time)
        x = x_windows_norm.permute(0, 2, 1)

        # CNN over time axis
        x = self.CNN(x)               # [B, cnn_channels, W]

        # Flatten time and channels
        x = x.flatten(start_dim=1)    # [B, cnn_channels * W]

        return x

    def forward(self, history, generate_steps=1):
        """
        Autoregressive generation — same sliding window as AutoregressionMLP.
        history: [Batch, WindowSize, FrameDim]
        returns: [Batch, generate_steps, FrameDim]
        """
        batch_size = history.shape[0]
        generated  = []
        current    = history.clone()

        for _ in range(generate_steps):
            # Normalise all frames in the window
            flat      = current.reshape(-1, self.FrameDim)
            flat_norm = self.InputStats.Normalize(flat)
            x_norm    = flat_norm.reshape(batch_size, self.WindowSize, self.FrameDim)

            # CNN feature extraction
            cnn_features = self._extract_features(x_norm)  # [B, cnn_out_dim]

            # MLP prediction
            pred_norm  = self.MLP(cnn_features)             # [B, FrameDim]
            next_frame = self.OutputStats.Denormalize(pred_norm)

            generated.append(next_frame)

            # Slide the window — same as AutoregressionMLP
            current = torch.cat(
                [current[:, 1:, :], next_frame.unsqueeze(1)], dim=1
            )

        return torch.stack(generated, dim=1)  # [B, generate_steps, FrameDim]

    def learn(self, input_windows, target_frames, update_stats):
        """
        input_windows: [Batch, WindowSize, FrameDim]
        target_frames: [Batch, FrameDim]
        """
        B = input_windows.size(0)

        if update_stats:
            self.InputStats.Update(input_windows.reshape(-1, self.FrameDim))
            self.OutputStats.Update(target_frames)

        # Normalise all frames in window
        flat      = input_windows.reshape(-1, self.FrameDim)
        flat_norm = self.InputStats.Normalize(flat)
        x_norm    = flat_norm.reshape(B, self.WindowSize, self.FrameDim)

        # Normalise targets
        y_norm = self.OutputStats.Normalize(target_frames)

        # CNN + MLP
        cnn_features = self._extract_features(x_norm)
        prediction   = self.MLP(cnn_features)

        loss = Losses.MSE(prediction, y_norm)
        return {"MSE Loss": loss}
