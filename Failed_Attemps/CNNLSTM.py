# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torch.nn as nn
import torch.nn.functional as F
from ai4animation.AI import Losses, Stats
from ai4animation.AI.Library import Defaults


class Model(nn.Module):
    def __init__(
        self,
        frame_dim,
        window_size,
        cnn_channels=256,
        lstm_hidden=256,
        lstm_layers=2,
        dropout=Defaults.Dropout,
    ):
        super(Model, self).__init__()

        self.FrameDim = frame_dim
        self.WindowSize = window_size
        self.LSTMHidden = lstm_hidden
        self.LSTMLayers = lstm_layers

        self.InputDim = frame_dim       # per-frame input
        self.OutputDim = frame_dim      # per-frame output (one future frame)

        # Running stats operate on single frames, not the full flattened window
        self.InputStats = Stats.RunningStats(self.InputDim)
        self.OutputStats = Stats.RunningStats(self.OutputDim)

        # ------------------------------------------------------------------
        # CNN block
        # Operates on the time axis: [Batch, FrameDim, WindowSize]
        # Two conv layers extract local temporal patterns (acceleration,
        # curvature, rhythm) across consecutive frames.
        # kernel_size=3 with padding=1 keeps the time dimension intact so
        # the LSTM still sees WindowSize timesteps.
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
        # LSTM block
        # Input per timestep: cnn_channels (output of CNN)
        # Carries a hidden state that persists between forward() calls,
        # giving the model genuine long-term memory across many windows.
        # ------------------------------------------------------------------
        self.LSTM = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,   # expects [Batch, Time, Features]
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # ------------------------------------------------------------------
        # Output head
        # Projects the last LSTM timestep to a single predicted frame.
        # ------------------------------------------------------------------
        self.OutputHead = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ELU(),
            nn.Linear(lstm_hidden, frame_dim),
        )

        # Hidden state that persists between forward() calls during inference.
        # None means LSTM will initialise to zeros on first call.
        self.HiddenState = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def ResetMemory(self):
        """Call this at the start of a new sequence to wipe the LSTM state."""
        self.HiddenState = None

    def DetachMemory(self):
        """Detach hidden state from the compute graph (call between batches)."""
        if self.HiddenState is not None:
            h, c = self.HiddenState
            self.HiddenState = (h.detach(), c.detach())

    # ------------------------------------------------------------------
    # Forward pass
    # history: [Batch, WindowSize, FrameDim]  (3-D, NOT flattened)
    # generate_steps: how many future frames to autoregressively produce
    # Returns: [Batch, generate_steps, FrameDim]
    # ------------------------------------------------------------------
    def forward(self, history, generate_steps=1):
        batch_size = history.shape[0]
        generated_frames = []
        current_history = history.clone()

        for _ in range(generate_steps):
            # Normalize each frame independently
            # [Batch, WindowSize, FrameDim] -> normalize along last dim
            x = current_history.reshape(-1, self.FrameDim)
            x = self.InputStats.Normalize(x)
            x = x.reshape(batch_size, self.WindowSize, self.FrameDim)

            # CNN expects [Batch, Channels, Time]
            x = x.permute(0, 2, 1)         # [Batch, FrameDim, WindowSize]
            x = self.CNN(x)                 # [Batch, cnn_channels, WindowSize]
            x = x.permute(0, 2, 1)         # [Batch, WindowSize, cnn_channels]

            # LSTM: reuse hidden state from previous call for memory
            x, self.HiddenState = self.LSTM(x, self.HiddenState)

            # Take only the last timestep (most recent prediction)
            x = x[:, -1, :]                # [Batch, lstm_hidden]

            # Project to frame dimension and denormalize
            next_frame_norm = self.OutputHead(x)                        # [Batch, FrameDim]
            next_frame = self.OutputStats.Denormalize(next_frame_norm)  # [Batch, FrameDim]

            generated_frames.append(next_frame)

            # Slide the window: drop oldest frame, append prediction
            current_history = torch.cat(
                [current_history[:, 1:, :], next_frame.unsqueeze(1)], dim=1
            )

        return torch.stack(generated_frames, dim=1)  # [Batch, generate_steps, FrameDim]

    # ------------------------------------------------------------------
    # Training pass
    # input_windows:  [Batch, WindowSize, FrameDim]  (3-D)
    # target_frames:  [Batch, FrameDim]
    # ------------------------------------------------------------------
    def learn(self, input_windows, target_frames, update_stats):
        batch_size = input_windows.shape[0]

        if update_stats:
            # Update stats on flat frames
            self.InputStats.Update(input_windows.reshape(-1, self.FrameDim))
            self.OutputStats.Update(target_frames)

        # Normalize inputs frame-by-frame
        x = input_windows.reshape(-1, self.FrameDim)
        x = self.InputStats.Normalize(x)
        x = x.reshape(batch_size, self.WindowSize, self.FrameDim)

        # Normalize targets
        y = self.OutputStats.Normalize(target_frames)

        # CNN
        x = x.permute(0, 2, 1)
        x = self.CNN(x)
        x = x.permute(0, 2, 1)

        # LSTM — during training we do NOT reuse self.HiddenState across
        # batches to avoid cross-contamination; we pass None so it resets.
        x, _ = self.LSTM(x, None)
        x = x[:, -1, :]

        # Output head
        prediction = self.OutputHead(x)

        loss = Losses.MSE(prediction, y)
        return {"MSE Loss": loss}
