# LSTM block and model 

import torch
import torch.nn as nn
from ai4animation.AI.Library import Defaults, Losses
from ai4animation.AI.Library.Statistics import RunningStatistics


class LSTMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, future_steps, num_layers=2, dropout=0.0):
        super(LSTMBlock, self).__init__()
        self.FutureSteps = future_steps
        self.StepOutputDim = output_dim // future_steps
        
        self.InputProjection = nn.Sequential( # compresses input dimension to another dimension
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh() # used to avoid huge gradients causing overflow
        )
        
        self.StepEmbedding = nn.Parameter(torch.zeros(1, future_steps, hidden_dim)) # positional tag for each frame
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.LinearOutput = nn.Linear(hidden_dim, self.StepOutputDim) # decompresses back to the right dimensions

        nn.init.zeros_(self.LinearOutput.weight) # initialises weights with zero to avoid random numbers causing problems in standaone mode
        nn.init.zeros_(self.LinearOutput.bias)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        x_proj = self.InputProjection(x)

        x_seq = x_proj.unsqueeze(1).repeat(1, self.FutureSteps, 1)
        x_seq = x_seq + self.StepEmbedding.expand(batch_size, -1, -1)

        lstm_out, _ = self.lstm(x_seq)

        out = self.LinearOutput(lstm_out)

        return out.reshape(batch_size, -1)

class Model(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        future_steps=6,
        num_layers=2,
        dropout=Defaults.Dropout,
    ):
        super(Model, self).__init__()

        self.InputStatistics = RunningStatistics(input_dim)
        self.OutputStatistics = RunningStatistics(output_dim)

        self.Layers = LSTMBlock(
            input_dim, hidden_dim, output_dim, future_steps, num_layers, dropout
        )

    def input_dim(self):
        return self.InputStatistics.Dim

    def output_dim(self):
        return self.OutputStatistics.Dim

    def forward(self, x):
        squeezed = x.dim() == 1
        if squeezed:
            x = x.unsqueeze(0)

        z = self.InputStatistics.Normalize(x)
        z = self.Layers(z)
        y = self.OutputStatistics.Denormalize(z)
        
        return y.squeeze(0) if squeezed else y

    def learn(self, input, output, update_statistics):
        if update_statistics:
            self.InputStatistics.Update(input)
            self.OutputStatistics.Update(output)

        input_norm = self.InputStatistics.Normalize(input)
        output_norm = self.OutputStatistics.Normalize(output)
        
        prediction = self.Layers(input_norm)

        loss = Losses.MSE(prediction, output_norm)

        return {"Y": self.OutputStatistics.Denormalize(prediction)}, {"MSE Loss": loss}