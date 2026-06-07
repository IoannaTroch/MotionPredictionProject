# Flow Matching model (PROBABILISTIC) for autoregressive motion prediction.
#
# Erwthma 5: anti gia nteterministiko montelo (MLP / LSTM) xrhsimopoioume ena
# aplo pithanotiko montelo: Conditional Flow Matching.
#
# Idea:
#   Mathainoume ena dianysmatiko pedio taxuthtas  v_theta(x_t, t | condition)
#   pou metaferei thoryvo apo katanomh N(0,1)  ->  to epomeno kare ths kinhshs.
#   - condition = ta prohgoumena kare (window), flattened   [B, cond_dim]
#   - target    = to epomeno kare (raw h latent)            [B, target_dim]
#   - t         = xronos rohs sto [0,1]
#
# Ekpaideush (rectified / linear flow):
#   noise ~ N(0,1),  t ~ U(0,1)
#   x_t = (1 - t) * noise + t * target
#   stoxos taxuthtas:  v* = target - noise
#   loss = MSE( v_theta(x_t, t, condition) ,  target - noise )
#
# Deigmatolhpsia (sampling):
#   ksekiname apo x_0 = thoryvo N(0,1) kai oloklhrwnoume thn ODE  dx/dt = v_theta
#   apo t=0 ews t=1 me midpoint (RK2) vhmata -> pairnoume to neo kare.
#
# H DOMH TWN LAYERS akolouthei to MultiLayerPerceptron.py:
#   RunningStatistics gia kanonikopoihsh + LinearBlock (3 grammika layers).
# DEN einai LSTM kai DEN einai anadromiko (recurrent).

import torch
import torch.nn as nn

from ai4animation.AI.Library import Defaults, Losses
from ai4animation.AI.Library.Blocks import LinearBlock
from ai4animation.AI.Library.Statistics import RunningStatistics


class Model(nn.Module):
    def __init__(
        self,
        cond_dim,        # diastash sunthikhs = window_size * (frame_dim h latent_dim)
        target_dim,      # diastash auto pou paragoume = frame_dim (raw) h latent_dim
        hidden_dim,      # neurwnes sta krufa layers
        dropout=0.0,     # 0 sto flow gia kathari paliggrafhsh taxuthtas (statheri rohh)
        activation=Defaults.Activation,
        steps=10,        # vhmata oloklhrwshs ODE sto sampling
    ):
        super(Model, self).__init__()

        self.CondDim = cond_dim
        self.TargetDim = target_dim
        self.Steps = steps

        # statistika kanonikopoihshs (idia logikh me MultiLayerPerceptron.py:
        #   InputStatistics / OutputStatistics)
        self.CondStatistics = RunningStatistics(cond_dim)
        self.TargetStatistics = RunningStatistics(target_dim)

        # to diktuo taxuthtas. eisodos = [ x_t , t , condition ], eksodos = velocity
        # idia domh me to MLP: LinearBlock me 3 grammika layers
        self.Layers = LinearBlock(
            target_dim + 1 + cond_dim, hidden_dim, target_dim, dropout, activation
        )

    def input_dim(self):
        return self.CondStatistics.Dim

    def output_dim(self):
        return self.TargetStatistics.Dim

    # ---- velocity field: provlepei thn taxuthta sto kanonikopoihmeno target space ----
    def velocity(self, x_t, t, cond):
        return self.Layers(torch.cat((x_t, t, cond), dim=-1))

    # ---- ena midpoint (RK2) vhma oloklhrwshs ths ODE (idio me Networks/Flow.py) ----
    def step(self, x_t, t_start, t_end, cond):
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        dt = t_end - t_start
        return x_t + dt * self.velocity(
            x_t + dt / 2 * self.velocity(x_t, t_start, cond), t_start + dt / 2, cond
        )

    # ---- SAMPLING: ksekinaei apo thoryvo kai paragei to epomeno kare ----
    # condition: [B, cond_dim]  ->  epistrefei [B, target_dim] sto pragmatiko (denorm) space
    def forward(self, condition, noise=None, steps=None):
        steps = self.Steps if steps is None else steps
        cond = self.CondStatistics.Normalize(condition)

        if noise is None:
            # pithanotiko: kainourgios thoryvos kathe fora
            x_t = torch.randn(cond.shape[0], self.TargetDim, device=cond.device)
        else:
            x_t = noise

        timestamps = torch.linspace(0, 1.0, steps + 1, device=cond.device)
        for i in range(steps):
            x_t = self.step(x_t, timestamps[i], timestamps[i + 1], cond)

        return self.TargetStatistics.Denormalize(x_t)

    # ---- EKPAIDEUSH: conditional flow matching loss ----
    def learn(self, condition, target, update_statistics):
        if update_statistics:
            self.CondStatistics.Update(condition)
            self.TargetStatistics.Update(target)

        cond = self.CondStatistics.Normalize(condition)
        target = self.TargetStatistics.Normalize(target)

        # deigmatolhpsia thoryvou kai xronou
        noise = torch.randn_like(target)
        t = torch.rand(target.shape[0], 1, device=target.device)

        # grammikh diadromh anamesa se thoryvo (t=0) kai target (t=1)
        x_t = (1.0 - t) * noise + t * target

        # to montelo provlepei thn taxuthta
        v_pred = self.velocity(x_t, t, cond)

        # h pragmatikh taxuthta ths grammikhs diadromhs einai (target - noise)
        v_target = target - noise

        loss = Losses.MSE(v_pred, v_target)

        # gia symvatothta me to training loop (idio interface me VAE: {"Y","Z"})
        with torch.no_grad():
            sample = self.TargetStatistics.Denormalize(x_t)

        return {"Y": sample, "Z": v_pred}, {"MSE Loss": loss}
