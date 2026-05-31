# second implementation of LSTM for use with SequencePrediction
import torch
import torch.nn as nn
from ai4animation.AI.Library import Defaults, Losses
from ai4animation.AI.Library.Statistics import RunningStatistics

class LSTMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, future_steps, num_layers=2):
        super(LSTMBlock, self).__init__()
        self.FutureSteps = future_steps
        self.StepOutputDim = output_dim // future_steps
        
        self.InputProjection = nn.Linear(input_dim, hidden_dim)
        # self.InputProjection = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.Tanh()
        # )
        
        self.StepEmbedding = nn.Parameter(torch.randn(1, future_steps, hidden_dim) * 0.01)
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        self.LinearOutput = nn.Linear(hidden_dim, self.StepOutputDim)
        
        # nn.init.zeros_(self.LinearOutput.weight)
        # nn.init.zeros_(self.LinearOutput.bias)
               
    def forward(self, x):
        batch_size = x.shape[0]
        
        x_proj = torch.relu(self.InputProjection(x))
        x_seq = x_proj.unsqueeze(1).repeat(1, self.FutureSteps, 1)
        x_seq = x_seq + self.StepEmbedding.expand(batch_size, -1, -1)
        
        lstm_out, _ = self.lstm(x_seq)
        
        out = self.LinearOutput(lstm_out)

###
        # B = (self.StepOutputDim - 4) // 9
        
        # root_pos = out[:, :, 0:2].reshape(batch_size, -1)           
        # root_fwd = out[:, :, 2:4].reshape(batch_size, -1)           
        # m_pos    = out[:, :, 4 : 4 + B*3].reshape(batch_size, -1)   
        # m_rot_z  = out[:, :, 4 + B*3 : 4 + B*6].reshape(batch_size, -1)
        # m_rot_y  = out[:, :, 4 + B*6 : 4 + B*9].reshape(batch_size, -1)
        
        # return torch.cat([root_pos, root_fwd, m_pos, m_rot_z, m_rot_y], dim=1)
        
        return out.reshape(batch_size, -1)


class Model(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        future_steps,
        num_layers=2,
        dropout=Defaults.Dropout,
        activation=Defaults.Activation,
    ):
        super(Model, self).__init__()

        self.InputStatistics = RunningStatistics(input_dim)
        self.OutputStatistics = RunningStatistics(output_dim)

        self.Layers = LSTMBlock(
            input_dim, hidden_dim, output_dim, future_steps, num_layers
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