# conditional flow matching model implementation for autoregressive motion prediction
# the model learns a vector field v_theta(x_t, t|condition) 
#   - condition = window with past frames, flattened 
#   - target    = next frame (raw or latent)            
#   - t         = time [0,1]

## this implementation uses Adaptive Layer Normalisation (AdaLN)
# based on https://github.com/facebookresearch/DiT/tree/main 

import math
import torch
import torch.nn as nn

from ai4animation.AI.Library import Defaults, Losses
from ai4animation.AI.Library.Statistics import RunningStatistics

class SinusoidalPositionEmbeddings(nn.Module): # adds time embeddings 
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class AdaLNResidualBlock(nn.Module): # residual block with AdaLN: Adaptive Layer Normalisation
    def __init__(self, dim, time_emb_dim, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.linear1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim, dim)
        
        # AdaLN Modulator: takes the time embedding and predicts 
        # shift, scale, and a gate for the skip connection
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 3 * dim, bias=True)
        )
        
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, t_emb):
        shift, scale, gate = self.adaLN_modulation(t_emb).chunk(3, dim=1)

        h = self.norm1(x)
        h = h * (1 + scale) + shift

        h = self.linear1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.linear2(h)
        
        return x + gate * h # gated skip connection -- gate starts at 0 and slowsly learns to let data through


class Model(nn.Module):
    def __init__(
        self,
        cond_dim,        
        target_dim,      
        hidden_dim=512,      
        dropout=0.1,     
        steps=10,        
    ):
        super(Model, self).__init__()

        self.CondDim = cond_dim
        self.TargetDim = target_dim
        self.Steps = steps

        self.CondStatistics = RunningStatistics(cond_dim)
        self.TargetStatistics = RunningStatistics(target_dim)

        self.TimeDim = 256 # higher time dimension gives better results
        self.TimeMLP = nn.Sequential(
            SinusoidalPositionEmbeddings(self.TimeDim),
            nn.Linear(self.TimeDim, self.TimeDim),
            nn.GELU()
        )

        in_dim = target_dim + cond_dim 
        
        self.InputLayer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.ResBlocks = nn.ModuleList([
            AdaLNResidualBlock(hidden_dim, self.TimeDim, dropout),
            AdaLNResidualBlock(hidden_dim, self.TimeDim, dropout),
            AdaLNResidualBlock(hidden_dim, self.TimeDim, dropout)
        ])
        
        self.OutputLayer = nn.Linear(hidden_dim, target_dim)

    def input_dim(self):
        return self.CondStatistics.Dim

    def output_dim(self):
        return self.TargetStatistics.Dim

    def velocity(self, x_t, t, cond):
        t_emb = self.TimeMLP(t)
        
        x = torch.cat((x_t, cond), dim=-1)
        
        x = self.InputLayer(x)
        
        for block in self.ResBlocks:
            x = block(x, t_emb)
            
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

        cond_noise_level = 0.02
        cond = cond + torch.randn_like(cond) * cond_noise_level

        noise = torch.randn_like(target)
        t = torch.rand(target.shape[0], 1, device=target.device)

        x_t = (1.0 - t) * noise + t * target
        v_pred = self.velocity(x_t, t, cond)
        v_target = target - noise

        loss = Losses.MSE(v_pred, v_target)

        with torch.no_grad():
            sample = self.TargetStatistics.Denormalize(x_t)

        return {"Y": sample, "Z": v_pred}, {"MSE Loss": loss}

####### simple implementation -- unsatisfying loss and behaviour #########
# import torch
# import torch.nn as nn

# from ai4animation.AI.Library import Defaults, Losses
# from ai4animation.AI.Library.Blocks import LinearBlock
# from ai4animation.AI.Library.Statistics import RunningStatistics


# class Model(nn.Module):
#     def __init__(
#         self,
#         cond_dim,        # diastash sunthikhs = window_size * (frame_dim h latent_dim)
#         target_dim,      # diastash auto pou paragoume = frame_dim (raw) h latent_dim
#         hidden_dim,      # neurwnes sta krufa layers
#         dropout=0.0,     # 0 sto flow gia kathari paliggrafhsh taxuthtas (statheri rohh)
#         activation=Defaults.Activation,
#         steps=10,        # vhmata oloklhrwshs ODE sto sampling
#     ):
#         super(Model, self).__init__()

#         self.CondDim = cond_dim
#         self.TargetDim = target_dim
#         self.Steps = steps

#         # statistika kanonikopoihshs (idia logikh me MultiLayerPerceptron.py:
#         #   InputStatistics / OutputStatistics)
#         self.CondStatistics = RunningStatistics(cond_dim)
#         self.TargetStatistics = RunningStatistics(target_dim)

#         # to diktuo taxuthtas. eisodos = [ x_t , t , condition ], eksodos = velocity
#         # idia domh me to MLP: LinearBlock me 3 grammika layers
#         self.Layers = LinearBlock(
#             target_dim + 1 + cond_dim, hidden_dim, target_dim, dropout, activation
#         )

#     def input_dim(self):
#         return self.CondStatistics.Dim

#     def output_dim(self):
#         return self.TargetStatistics.Dim

#     # ---- velocity field: provlepei thn taxuthta sto kanonikopoihmeno target space ----
#     def velocity(self, x_t, t, cond):
#         return self.Layers(torch.cat((x_t, t, cond), dim=-1))

#     # ---- ena midpoint (RK2) vhma oloklhrwshs ths ODE (idio me Networks/Flow.py) ----
#     def step(self, x_t, t_start, t_end, cond):
#         t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
#         dt = t_end - t_start
#         return x_t + dt * self.velocity(
#             x_t + dt / 2 * self.velocity(x_t, t_start, cond), t_start + dt / 2, cond
#         )

#     # ---- SAMPLING: ksekinaei apo thoryvo kai paragei to epomeno kare ----
#     # condition: [B, cond_dim]  ->  epistrefei [B, target_dim] sto pragmatiko (denorm) space
#     def forward(self, condition, noise=None, steps=None):
#         steps = self.Steps if steps is None else steps
#         cond = self.CondStatistics.Normalize(condition)

#         if noise is None:
#             # pithanotiko: kainourgios thoryvos kathe fora
#             x_t = torch.randn(cond.shape[0], self.TargetDim, device=cond.device)
#         else:
#             x_t = noise

#         timestamps = torch.linspace(0, 1.0, steps + 1, device=cond.device)
#         for i in range(steps):
#             x_t = self.step(x_t, timestamps[i], timestamps[i + 1], cond)

#         return self.TargetStatistics.Denormalize(x_t)

#     # ---- EKPAIDEUSH: conditional flow matching loss ----
#     def learn(self, condition, target, update_statistics):
#         if update_statistics:
#             self.CondStatistics.Update(condition)
#             self.TargetStatistics.Update(target)

#         cond = self.CondStatistics.Normalize(condition)
#         target = self.TargetStatistics.Normalize(target)

#         # deigmatolhpsia thoryvou kai xronou
#         noise = torch.randn_like(target)
#         t = torch.rand(target.shape[0], 1, device=target.device)

#         # grammikh diadromh anamesa se thoryvo (t=0) kai target (t=1)
#         x_t = (1.0 - t) * noise + t * target

#         # to montelo provlepei thn taxuthta
#         v_pred = self.velocity(x_t, t, cond)

#         # h pragmatikh taxuthta ths grammikhs diadromhs einai (target - noise)
#         v_target = target - noise

#         loss = Losses.MSE(v_pred, v_target)

#         # gia symvatothta me to training loop (idio interface me VAE: {"Y","Z"})
#         with torch.no_grad():
#             sample = self.TargetStatistics.Denormalize(x_t)

#         return {"Y": sample, "Z": v_pred}, {"MSE Loss": loss}

########## model with added time embeddings and a residual layer ############
##### a little higher loss but a lot better behaviour than the previous implementation
# import math
# import torch
# import torch.nn as nn

# from ai4animation.AI.Library import Defaults, Losses
# from ai4animation.AI.Library.Statistics import RunningStatistics

# class SinusoidalPositionEmbeddings(nn.Module): # time embeddings to convert time to a 64D vector
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, time):
#         device = time.device
#         half_dim = self.dim // 2
#         embeddings = math.log(10000) / (half_dim - 1)
#         embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
#         embeddings = time * embeddings[None, :]
#         embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
#         return embeddings

# class ResidualBlock(nn.Module): # residual block 
#     def __init__(self, dim, dropout=0.1):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, dim),
#             nn.LayerNorm(dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(dim, dim),
#             nn.LayerNorm(dim)
#         )

#     def forward(self, x):
#         return x + self.net(x) # Skip connection

# class Model(nn.Module):
#     def __init__(
#         self,
#         cond_dim,        
#         target_dim,      
#         hidden_dim=512,      
#         dropout=0.1,     
#         steps=10,        
#     ):
#         super(Model, self).__init__()

#         self.CondDim = cond_dim
#         self.TargetDim = target_dim
#         self.Steps = steps

#         self.CondStatistics = RunningStatistics(cond_dim)
#         self.TargetStatistics = RunningStatistics(target_dim)

#         self.TimeDim = 64
#         self.TimeMLP = nn.Sequential(
#             SinusoidalPositionEmbeddings(self.TimeDim),
#             nn.Linear(self.TimeDim, self.TimeDim),
#             nn.GELU()
#         )

#         # Changed the + 1 to + self.TimeDim
#         in_dim = target_dim + self.TimeDim + cond_dim
        
#         self.InputLayer = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.GELU()
#         )
        
#         self.ResBlocks = nn.Sequential(
#             ResidualBlock(hidden_dim, dropout),
#             ResidualBlock(hidden_dim, dropout),
#             ResidualBlock(hidden_dim, dropout)
#         )
        
#         self.OutputLayer = nn.Linear(hidden_dim, target_dim)

#     def input_dim(self):
#         return self.CondStatistics.Dim

#     def output_dim(self):
#         return self.TargetStatistics.Dim

#     def velocity(self, x_t, t, cond):
#         # Push 't' through the embedding layer before concatenating
#         t_emb = self.TimeMLP(t)
#         x = torch.cat((x_t, t_emb, cond), dim=-1)
        
#         x = self.InputLayer(x)
#         x = self.ResBlocks(x)
#         return self.OutputLayer(x)

#     def step(self, x_t, t_start, t_end, cond):
#         t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
#         dt = t_end - t_start
#         return x_t + dt * self.velocity(
#             x_t + dt / 2 * self.velocity(x_t, t_start, cond), t_start + dt / 2, cond
#         )

#     def forward(self, condition, noise=None, steps=None):
#         # Note: If your animations look jittery, bump steps up to 30 or 50 during inference
#         steps = self.Steps if steps is None else steps
#         cond = self.CondStatistics.Normalize(condition)

#         if noise is None:
#             x_t = torch.randn(cond.shape[0], self.TargetDim, device=cond.device)
#         else:
#             x_t = noise

#         timestamps = torch.linspace(0, 1.0, steps + 1, device=cond.device)
#         for i in range(steps):
#             x_t = self.step(x_t, timestamps[i], timestamps[i + 1], cond)

#         return self.TargetStatistics.Denormalize(x_t)

#     def learn(self, condition, target, update_statistics):
#         if update_statistics:
#             self.CondStatistics.Update(condition)
#             self.TargetStatistics.Update(target)

#         cond = self.CondStatistics.Normalize(condition)
#         target = self.TargetStatistics.Normalize(target)

#         noise = torch.randn_like(target)
#         t = torch.rand(target.shape[0], 1, device=target.device)

#         x_t = (1.0 - t) * noise + t * target
#         v_pred = self.velocity(x_t, t, cond)
#         v_target = target - noise

#         loss = Losses.MSE(v_pred, v_target)

#         with torch.no_grad():
#             sample = self.TargetStatistics.Denormalize(x_t)

#         return {"Y": sample, "Z": v_pred}, {"MSE Loss": loss}

