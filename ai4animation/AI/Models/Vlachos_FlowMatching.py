import torch
import torch.nn as nn

from ai4animation.AI.Library import Defaults, Losses
from ai4animation.AI.Library.Statistics import RunningStatistics

# -------------------------------------------------------------
# ΝΕΟ: Residual Block για ομαλότητα και "Βαθιά" Εκπαίδευση
# -------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
    def forward(self, x):
        return x + self.net(x) # Skip connection

class Model(nn.Module):
    def __init__(
        self,
        cond_dim,        
        target_dim,      
        hidden_dim=512,      
        dropout=0.1,     # Πλέον χρησιμοποιούμε Dropout 10% για σταθερότητα
        steps=10,        
    ):
        super(Model, self).__init__()

        self.CondDim = cond_dim
        self.TargetDim = target_dim
        self.Steps = steps

        self.CondStatistics = RunningStatistics(cond_dim)
        self.TargetStatistics = RunningStatistics(target_dim)

        # -------------------------------------------------------------
        # ΝΕΟ: Deep Residual Architecture αντί για απλό LinearBlock
        # -------------------------------------------------------------
        in_dim = target_dim + 1 + cond_dim
        
        self.InputLayer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 3 Block σημαίνουν 6 κρυφά επίπεδα με εξαιρετική ροή πληροφορίας!
        self.ResBlocks = nn.Sequential(
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout)
        )
        
        self.OutputLayer = nn.Linear(hidden_dim, target_dim)

    def input_dim(self):
        return self.CondStatistics.Dim

    def output_dim(self):
        return self.TargetStatistics.Dim

    def velocity(self, x_t, t, cond):
        x = torch.cat((x_t, t, cond), dim=-1)
        x = self.InputLayer(x)
        x = self.ResBlocks(x)
        return self.OutputLayer(x)

    def step(self, x_t, t_start, t_end, cond):
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        dt = t_end - t_start
        return x_t + dt * self.velocity(
            x_t + dt / 2 * self.velocity(x_t, t_start, cond), t_start + dt / 2, cond
        )

    def forward(self, condition, noise=None, steps=None):
        steps = self.Steps if steps is None else steps
        cond = self.CondStatistics.Normalize(condition)

        if noise is None:
            x_t = torch.randn(cond.shape[0], self.TargetDim, device=cond.device)
        else:
            x_t = noise

        timestamps = torch.linspace(0, 1.0, steps + 1, device=cond.device)
        for i in range(steps):
            x_t = self.step(x_t, timestamps[i], timestamps[i + 1], cond)

        return self.TargetStatistics.Denormalize(x_t)

    def learn(self, condition, target, update_statistics):
        if update_statistics:
            self.CondStatistics.Update(condition)
            self.TargetStatistics.Update(target)

        cond = self.CondStatistics.Normalize(condition)
        target = self.TargetStatistics.Normalize(target)

        noise = torch.randn_like(target)
        t = torch.rand(target.shape[0], 1, device=target.device)

        x_t = (1.0 - t) * noise + t * target
        v_pred = self.velocity(x_t, t, cond)
        v_target = target - noise

        loss = Losses.MSE(v_pred, v_target)

        with torch.no_grad():
            sample = self.TargetStatistics.Denormalize(x_t)

        return {"Y": sample, "Z": v_pred}, {"MSE Loss": loss}