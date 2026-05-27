import torch
import torch.nn as nn
from ai4animation.AI.Library.Statistics import RunningStatistics
from ai4animation.AI.Library import Losses

class TransformerAutoencoder(nn.Module):
    def __init__(self, feature_dim, latent_dim=32):
        super().__init__()
        
        self.Statistics = RunningStatistics(feature_dim)
        
        #kathe osto exei 12 features
        self.bone_features = 12
        self.num_bones = feature_dim // self.bone_features

        #positional encoding, dinw etiketes se kathe osto
        self.pos_embedding_encoder = nn.Parameter(torch.randn(1, self.num_bones, self.bone_features) * 0.02)
        self.pos_embedding_decoder = nn.Parameter(torch.randn(1, self.num_bones, self.bone_features) * 0.02)

        #encoder ena transformer layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.bone_features,
            nhead=4,
            dim_feedforward=128, # Το αυξήσαμε λίγο για να του δώσουμε "χώρο" να σκεφτεί
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        #vazw kai ena mlp meta apo to transformer layer
        self.encoder_linear = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim)
        )

        #decoder
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, feature_dim)
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.bone_features,
            nhead=4,
            dim_feedforward=128,
            activation="gelu",
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=2)

    def encode(self, x):
        batch_size = x.shape[0]
        seq = x.reshape(batch_size, -1, self.bone_features)
        
        #vazw positional encoding
        seq = seq + self.pos_embedding_encoder
        
        attended_seq = self.transformer_encoder(seq)
        
        flat = attended_seq.reshape(batch_size, -1)
        return self.encoder_linear(flat)

    def decode(self, z):
        batch_size = z.shape[0]
        
        flat = self.decoder_linear(z)
        
        seq = flat.reshape(batch_size, -1, self.bone_features)
        
        seq = seq + self.pos_embedding_decoder
        
        refined_seq = self.transformer_decoder(seq)
        
        return refined_seq.reshape(batch_size, -1)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        x = self.Statistics.Normalize(x)
        z = self.encode(x)
        y_pred = self.decode(z)
        return self.Statistics.Denormalize(y_pred)

    def learn(self, features, update_statistics):
        if update_statistics:
            self.Statistics.Update(features)

        norm_features = self.Statistics.Normalize(features)
        
        latent = self.encode(norm_features)
        prediction = self.decode(latent)

        loss = Losses.MSE(prediction, norm_features)

        reconstruction = self.Statistics.Denormalize(prediction)
        return {"Y": reconstruction, "Z": latent}, {"MSE Loss": loss}