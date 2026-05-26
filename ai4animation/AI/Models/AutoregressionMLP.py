# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torch.nn as nn
from ai4animation.AI import Losses, Modules, Stats
from ai4animation.AI.Library import Defaults

class Model(nn.Module):
    def __init__(self, frame_dim, window_size, hidden_dim, dropout=Defaults.Dropout):
        super(Model, self).__init__()

        self.FrameDim = frame_dim
        self.WindowSize = window_size
        
        # input dimension = (number of past frames) x (features per frame)
        self.InputDim = window_size * frame_dim
        self.OutputDim = frame_dim # predicts one future frame

        self.InputStats = Stats.RunningStats(self.InputDim)
        self.OutputStats = Stats.RunningStats(self.OutputDim)

        self.Layers = Modules.LinearEncoder(self.InputDim, hidden_dim, self.OutputDim, dropout)

    def input_dim(self):
        return self.InputStats.Dim

    def output_dim(self):
        return self.OutputStats.Dim

    def forward(self, history, generate_steps=1):
        batch_size = history.shape[0]
        generated_frames = []
        
        # clones history to prevent modifying original data tensors
        current_history = history.clone()

        for _ in range(generate_steps):
            # flattens the sliding window: [Batch, WindowSize, FrameDim] -> [Batch, InputDim]
            x = current_history.reshape(batch_size, self.InputDim)
            
            x = self.InputStats.Normalize(x)
            z = self.Layers(x)
            next_frame = self.OutputStats.Denormalize(z) # shape: [Batch, FrameDim]
            
            generated_frames.append(next_frame)

            # slides the window -- deletes older frame and adds the prediction
            current_history = torch.cat([current_history[:, 1:, :], next_frame.unsqueeze(1)], dim=1)

        # stacks predictions along the time axis: [Batch, generate_steps, FrameDim]
        return torch.stack(generated_frames, dim=1)

    def learn(self, input_windows, target_frames, update_stats):
        if update_stats:
            self.InputStats.Update(input_windows)
            self.OutputStats.Update(target_frames)

        input_windows = self.InputStats.Normalize(input_windows)
        target_frames = self.OutputStats.Normalize(target_frames)
        
        prediction = self.Layers(input_windows)
        loss = Losses.MSE(prediction, target_frames)

        return {"MSE Loss": loss}