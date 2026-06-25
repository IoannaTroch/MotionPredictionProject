# Copyright (c) Meta Platforms, Inc. and affiliates.
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
        cnn_channels=128,
        hidden_dim=512,
        dropout=Defaults.Dropout,
    ):
        super(Model, self).__init__()

        self.FrameDim = frame_dim
        self.WindowSize = window_size
        self.OutputDim = frame_dim

        # Running stats on single frames (not the flattened window)
        self.InputStats = Stats.RunningStats(frame_dim)
        self.OutputStats = Stats.RunningStats(frame_dim)

        # ------------------------------------------------------------------
        # CNN block
        # Input:  [Batch, FrameDim, WindowSize]  (channels=FrameDim, length=WindowSize)
        # Two conv layers with kernel_size=3 and padding=1 so the time
        # dimension stays at WindowSize throughout.
        # Each layer learns local temporal patterns across neighbouring frames
        # (velocity, acceleration, joint curvature, rhythm).
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

        # ------------------------------------------------------------------
        # Global average pooling over the time axis
        # Collapses [Batch, cnn_channels, WindowSize] -> [Batch, cnn_channels]
        # This gives the MLP a single fixed-size feature vector regardless
        # of window size, and is more parameter-efficient than flattening.
        # ------------------------------------------------------------------
        self.Pool = nn.AdaptiveAvgPool1d(1)

        # ------------------------------------------------------------------
        # MLP block — same LinearEncoder structure used across the codebase
        # Input:  cnn_channels  (pooled CNN output)
        # Output: frame_dim     (one predicted future frame)
        # ------------------------------------------------------------------
        self.MLP = Modules.LinearEncoder(
            cnn_channels, hidden_dim, frame_dim, dropout
        )

    # ------------------------------------------------------------------
    # forward
    # history: [Batch, WindowSize, FrameDim]
    # Returns: [Batch, generate_steps, FrameDim]
    # ------------------------------------------------------------------
    def forward(self, history, generate_steps=1):
        batch_size = history.shape[0]
        generated_frames = []
        current_history = history.clone()

        for _ in range(generate_steps):
            # Normalize every frame in the window individually
            x = current_history.reshape(-1, self.FrameDim)
            x = self.InputStats.Normalize(x)
            x = x.reshape(batch_size, self.WindowSize, self.FrameDim)

            # CNN expects [Batch, Channels, Time]
            x = x.permute(0, 2, 1)             # [Batch, FrameDim, WindowSize]
            x = self.CNN(x)                     # [Batch, cnn_channels, WindowSize]

            # Pool across time -> [Batch, cnn_channels, 1] -> [Batch, cnn_channels]
            x = self.Pool(x).squeeze(-1)

            # MLP
            x = self.MLP(x)                     # [Batch, FrameDim]

            next_frame = self.OutputStats.Denormalize(x)
            generated_frames.append(next_frame)

            # Slide the window
            current_history = torch.cat(
                [current_history[:, 1:, :], next_frame.unsqueeze(1)], dim=1
            )

        return torch.stack(generated_frames, dim=1)  # [Batch, generate_steps, FrameDim]

    # ------------------------------------------------------------------
    # learn
    # input_windows: [Batch, WindowSize, FrameDim]
    # target_frames: [Batch, FrameDim]
    # ------------------------------------------------------------------
    def learn(self, input_windows, target_frames, update_stats):
        batch_size = input_windows.shape[0]

        if update_stats:
            self.InputStats.Update(input_windows.reshape(-1, self.FrameDim))
            self.OutputStats.Update(target_frames)

        # Normalize inputs frame-by-frame
        x = input_windows.reshape(-1, self.FrameDim)
        x = self.InputStats.Normalize(x)
        x = x.reshape(batch_size, self.WindowSize, self.FrameDim)

        # Normalize targets
        y = self.OutputStats.Normalize(target_frames)

        # CNN + pool
        x = x.permute(0, 2, 1)
        x = self.CNN(x)
        x = self.Pool(x).squeeze(-1)

        # MLP
        prediction = self.MLP(x)

        loss = Losses.MSE(prediction, y)
        return {"MSE Loss": loss}
