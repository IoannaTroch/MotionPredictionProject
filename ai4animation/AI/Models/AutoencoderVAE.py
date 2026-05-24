import torch
import torch.nn as nn
from ai4animation.AI.Library.Statistics import RunningStatistics
from ai4animation.AI.Library import Losses

class VAEAutoencoder(nn.Module):
    def __init__(self, feature_dim, latent_dim=32):
        super().__init__()
        
        self.Statistics = RunningStatistics(feature_dim)

        self.EncoderBody = nn.Sequential(#idios encoder me prin
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        
       
        #VAE layers, kanw to montelo na provlepei katanomh
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        self.Decoder = nn.Sequential(#idios decoder me prin
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, feature_dim) # Καθαρά νούμερα στην έξοδο
        )

    def reparameterize(self, mu, logvar):#sunarthsh gia na prosthesw thoruvo
        if self.training:#an eimai se training mode
            #kanw to logvar tupikh apoklish
            std = torch.exp(0.5 * logvar) 
            #vazw tuxaio thoryvo sth mesh timh gia na mporesw na kanw train to VAE
            eps = torch.randn_like(std)   
            return mu + eps * std
        else:
            #an eimai se validation krataw to mu xwris to thoruvo
            return mu 

    def forward(self, x):
        #auth h sunarthsh kaleite otan zwgrafizei to 3d montelo
        x = self.Statistics.Normalize(x)
        hidden = self.EncoderBody(x)
        mu = self.fc_mu(hidden)
        #sto forward xrhsimopoiw katharh mesh timh gt thelw stathero optiko apotelesma
        y = self.Decoder(mu)
        return self.Statistics.Denormalize(y)

    def learn(self, features, update_statistics):
        #auth h sunarthsh xrhsimopoieite gia thn ekpaideush

        #kanw update statistika tou dataset (ta kanonikopoiw)
        if update_statistics:
            self.Statistics.Update(features)

        norm_features = self.Statistics.Normalize(features)
        
        #pernaw apo encoder gia na vrw katanomh
        hidden = self.EncoderBody(norm_features)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)

        #edw prosthetoume tuxaio thoruvo
        latent = self.reparameterize(mu, logvar)
        
        #penaw decoder
        prediction = self.Decoder(latent)

    
        #upologizw reconstruction loss + KLD Loss
        #MSE reconstruction Loss
        recon_loss = Losses.MSE(prediction, norm_features)
        
        #koitaw poso to diktuo apokleinei apo kanonikh katanomh N(0,1)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        #kanonikopoiw klimaka kld_loss wste na einai idia me to MSE LOSS
        kld_loss = kld_loss / (norm_features.size(0) * norm_features.size(1))
        
        #b kathorizei poso austhroi eimaste gia na fugei apo to N(0,1)
        beta = 0.005 
        
        total_loss = recon_loss + beta * kld_loss

        reconstruction = self.Statistics.Denormalize(prediction)
        
        return {"Y": reconstruction, "Z": latent}, {"MSE Loss": total_loss}
