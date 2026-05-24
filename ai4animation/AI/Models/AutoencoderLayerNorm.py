import torch.nn as nn
from ai4animation.AI.Library.Statistics import RunningStatistics
from ai4animation.AI.Library import Losses

class EnchancedAutoencoder(nn.Module):
    def __init__(self, feature_dim, latent_dim=32):
        super().__init__()
        
        self.Statistics = RunningStatistics(feature_dim)

        #encoder
        self.Encoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, latent_dim) 
        )

        #decoder
        self.Decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, feature_dim) 
        )

    def forward(self, x):
        #auth trexei otan optikopoiw sto 3d dld kanei Draw()
        x = self.Statistics.Normalize(x)
        z = self.Encoder(x)
        z = self.Decoder(z)
        y = self.Statistics.Denormalize(z)
        return y

    def learn(self, features, update_statistics):
        #auth trexei kata to training
        if update_statistics:
            self.Statistics.Update(features)

        norm_features = self.Statistics.Normalize(features)
        
        latent = self.Encoder(norm_features)
        prediction = self.Decoder(latent)

        loss = Losses.MSE(prediction, norm_features)

        reconstruction = self.Statistics.Denormalize(prediction)
        return {"Y": reconstruction, "Z": latent}, {"MSE Loss": loss}