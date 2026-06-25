import torch
import torch.nn as nn
from ai4animation.AI.Library import Defaults, Losses
from ai4animation.AI.Library.Statistics import RunningStatistics

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
        return x + self.net(x)

class Model(nn.Module):
    def __init__(self, cond_dim, target_dim, hidden_dim=512, dropout=0.1, steps=10):
        super(Model, self).__init__()
        self.CondDim = cond_dim
        self.TargetDim = target_dim
        self.Steps = steps
        
        self.CondStatistics = RunningStatistics(cond_dim)
        self.TargetStatistics = RunningStatistics(target_dim)

        in_dim = target_dim + 1 + cond_dim
        self.InputLayer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.ResBlocks = nn.Sequential(
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout)
        )
        self.OutputLayer = nn.Linear(hidden_dim, target_dim)

    def input_dim(self): return self.CondStatistics.Dim
    def output_dim(self): return self.TargetStatistics.Dim

    def velocity(self, x_t, t, cond):
        x = torch.cat((x_t, t, cond), dim=-1)
        x = self.InputLayer(x)
        x = self.ResBlocks(x)
        return self.OutputLayer(x)

    def step_cfg(self, x_t, t_start, t_end, cond, null_cond, guidance_scale):
        t_start_expand = t_start.view(1, 1).expand(x_t.shape[0], 1)
        dt = t_end - t_start

        if guidance_scale <= 1.0:
            # Απλή Επίλυση ODE
            v_t = self.velocity(x_t, t_start_expand, cond)
            x_mid = x_t + dt / 2 * v_t
            t_mid = t_start_expand + dt / 2
            v_mid = self.velocity(x_mid, t_mid, cond)
            return x_t + dt * v_mid
        else:
            # CFG Επίλυση: Μαγική ενίσχυση της συνθήκης!
            v_null_t = self.velocity(x_t, t_start_expand, null_cond)
            v_cond_t = self.velocity(x_t, t_start_expand, cond)
            v_t = v_null_t + guidance_scale * (v_cond_t - v_null_t)

            x_mid = x_t + dt / 2 * v_t
            t_mid = t_start_expand + dt / 2

            v_null_mid = self.velocity(x_mid, t_mid, null_cond)
            v_cond_mid = self.velocity(x_mid, t_mid, cond)
            v_mid = v_null_mid + guidance_scale * (v_cond_mid - v_null_mid)

            return x_t + dt * v_mid

    def forward(self, condition, null_condition=None, guidance_scale=1.0, noise=None, steps=None):
        steps = self.Steps if steps is None else steps
        cond = self.CondStatistics.Normalize(condition)

        if null_condition is not None:
            null_cond = self.CondStatistics.Normalize(null_condition)
        else:
            null_cond = cond

        if noise is None:
            x_t = torch.randn(cond.shape[0], self.TargetDim, device=cond.device)
        else:
            x_t = noise

        timestamps = torch.linspace(0, 1.0, steps + 1, device=cond.device)
        for i in range(steps):
            x_t = self.step_cfg(x_t, timestamps[i], timestamps[i + 1], cond, null_cond, guidance_scale)

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