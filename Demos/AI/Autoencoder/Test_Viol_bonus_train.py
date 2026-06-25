# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
from pathlib import Path
import numpy as np

import torch
import matplotlib.pyplot as plt
from ai4animation import (
    AI4Animation, DataSampler, Dataset, FeedTensor, MirrorModule, 
    MotionModule, RootModule, Tensor, Transform, Utility, Vector3
)
from ai4animation.AI.Models.AutoencoderVAE import VAEAutoencoder

from ai4animation.AI.Models.FlowMatchingViol import Model as FlowMatchingModel 

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/100style")
sys.path.insert(0, ASSETS_PATH) 
import Definitions

EPOCH_COUNT = 150
BATCH_SIZE = 32
FRAMERATE = 30
BONES = Definitions.FULL_BODY_NAMES
FRAME_DIM = 12 * len(BONES)
LATENT_DIM = 256

WINDOW_SIZE = 15 
NUM_STYLES = 3   
CONDITION_MULTIPLIER = 10 

class Program:
    def Start(self):
        Utility.SetSeed(23456)

        self.Dataset = Dataset(
            os.path.join(ASSETS_PATH, "Motions"),
            [
                lambda x: RootModule(x, Definitions.HipName, Definitions.LeftHipName, Definitions.RightHipName, Definitions.LeftShoulderName, Definitions.RightShoulderName, Definitions.NeckName),
                lambda x: MotionModule(x),
                lambda x: MirrorModule(x, Vector3.Axis.ZPositive, Vector3.Create(0, 0, 180)),
            ],
        )

        self.DataSampler = DataSampler(
            self.Dataset,
            framerate=FRAMERATE,
            batch_size=BATCH_SIZE,
            function=self.GetTrainingFeatures,
        )

        print("Loading pre-trained VAE...")
        if os.path.exists("vae_full_model_100style.pth"):
            self.VAE = torch.load("vae_full_model_100style.pth", weights_only=False)
            self.VAE = Tensor.ToDevice(self.VAE)
        else:
            raise Exception("Δεν βρέθηκε το vae_full_model.pth! Τρέξε πρώτα το ProgramVAE.py.")
            
        self.VAE.eval()
        for param in self.VAE.parameters():
            param.requires_grad = False

        condition_dim = (WINDOW_SIZE * LATENT_DIM) + (NUM_STYLES * CONDITION_MULTIPLIER)
        
        self.Network = Tensor.ToDevice(
            FlowMatchingModel(
                cond_dim=condition_dim,
                target_dim=LATENT_DIM,
                hidden_dim=1024,  #512,
                dropout=0.1,
                steps=10
            )
        )

        self.HistoryOffsets = torch.arange(-WINDOW_SIZE, 0) / FRAMERATE
        self.Trainer = self.Training()

    def Standalone(self): pass
    def Update(self):
        try: next(self.Trainer)
        except StopIteration: pass

    def EncodeBatch(self, raw_data, window_size):
        batch_size = raw_data.shape[0]
        raw_reshaped = raw_data.view(-1, FRAME_DIM)
        norm_raw = self.VAE.Statistics.Normalize(Tensor.ToDevice(raw_reshaped.clone().detach().float()))
        hidden = self.VAE.EncoderBody(norm_raw)
        mu = self.VAE.fc_mu(hidden)
        return mu.view(batch_size, window_size * LATENT_DIM)

    def Training(self):
        print("Splitting dataset into Training (80%) and Validation (20%)...")
        all_batches = list(self.DataSampler.SampleBatchesWithinMotions(1, EPOCH_COUNT))
        
        split_idx = int(0.8 * len(all_batches))
        train_batches, val_batches = all_batches[:split_idx], all_batches[split_idx:]
        total_train_samples = sum([b[1].shape[0] for b in train_batches])
        
        self.Optimizer = Utility.CosineAnnealingOptimizer(
            self.Network.parameters(), self.DataSampler.BatchSize, total_train_samples
        )
        
        train_losses, val_losses = [], []
        best_val_loss = float('inf')

        for epoch in range(1, EPOCH_COUNT + 1):
            print(f"\n--- Epoch {epoch}/{EPOCH_COUNT} ---")
            
            self.Network.train()
            epoch_train_loss = 0.0
            
            for i, batch_data in enumerate(train_batches):
                xBatch_raw, yBatch_raw, condition = batch_data
                
                with torch.no_grad():
                    xBatch_latent = self.EncodeBatch(xBatch_raw, WINDOW_SIZE)
                    x_1 = self.EncodeBatch(yBatch_raw, 1) 

                xBatch_latent = xBatch_latent + torch.randn_like(xBatch_latent) * 0.05
          

                if torch.rand(1).item() < (0.30 * (epoch / EPOCH_COUNT)):
                    xBatch_latent = torch.zeros_like(xBatch_latent)

                condition_amplified = condition.repeat(1, CONDITION_MULTIPLIER)

                if torch.rand(1).item() < 0.10:
                    condition_amplified = torch.zeros_like(condition_amplified)

                cond_input = torch.cat([xBatch_latent, condition_amplified], dim=1)

                _, loss_dict = self.Network.learn(cond_input, x_1, update_statistics=(epoch == 1))
                loss = sum(loss_dict.values()) if isinstance(loss_dict, dict) else loss_dict

                torch.nn.utils.clip_grad_norm_(self.Network.parameters(), max_norm=1.0)

                
                self.Optimizer.Update(x_1.shape[0], loss)
                epoch_train_loss += loss.item()
                
                print(f"Training Progress: {100 * (i + 1) / len(train_batches):.1f}%", end="\r")
                yield
                
            print(" " * 50, end="\r")
            avg_train_loss = epoch_train_loss / len(train_batches)
            train_losses.append(avg_train_loss)

            self.Network.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for i, batch_data in enumerate(val_batches):
                    xBatch_raw, yBatch_raw, condition = batch_data
                    xBatch_latent = self.EncodeBatch(xBatch_raw, WINDOW_SIZE)
                    x_1 = self.EncodeBatch(yBatch_raw, 1)
                    
                    cond_input = torch.cat([xBatch_latent, condition.repeat(1, CONDITION_MULTIPLIER)], dim=1)
                    
                    _, loss_dict = self.Network.learn(cond_input, x_1, update_statistics=False)
                    loss = sum(loss_dict.values()).item() if isinstance(loss_dict, dict) else loss_dict.item()
                    
                    epoch_val_loss += loss
                    yield
                    
            avg_val_loss = epoch_val_loss / len(val_batches)
            val_losses.append(avg_val_loss)
            print(f"Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(">>> New best validation loss! <<<")

            self.PlotTrainVal(train_losses, val_losses, epoch)

        plt.ioff()
        plt.savefig("loss_history_FlowMatching.png", dpi=300, bbox_inches='tight')
        torch.save(self.Network, "viol_100style_bonus_flow_matching_full.pth")
        print("\n>>> Το Flow Matching (CFG) μοντέλο αποθηκεύτηκε επιτυχώς! <<<")
        plt.show()

    def PlotTrainVal(self, train_losses, val_losses, epoch):
        plt.ion()
        plt.clf()
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='orange')
        plt.title(f'Flow Matching Vector Field Loss ({epoch}/{EPOCH_COUNT})')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.pause(0.01)

    def ExtractFrameFeatures(self, motion, timestamps, mirrored, root):
        transforms = Transform.TransformationFrom(motion.GetBoneTransformations(timestamps, BONES, mirrored=mirrored), root.reshape(-1, 1, 4, 4))
        velocities = Vector3.DirectionFrom(motion.GetBoneVelocities(timestamps, BONES, mirrored=mirrored), root.reshape(-1, 1, 4, 4))
        inputs = FeedTensor("Frame", (len(timestamps), FRAME_DIM))
        inputs.Feed(Transform.GetPosition(transforms))
        inputs.Feed(Transform.GetAxisZ(transforms))
        inputs.Feed(Transform.GetAxisY(transforms))
        inputs.Feed(velocities)
        return inputs.GetTensor()

    def GetTrainingFeatures(self, batch):
        motion, timestamps = batch
        if isinstance(timestamps, np.ndarray): timestamps = torch.from_numpy(timestamps)
        mirrored = Tensor.RandomBool()

        root = Tensor.Inverse(motion.GetModule(RootModule).GetTransforms(timestamps, mirrored=mirrored))
        frames = self.ExtractFrameFeatures(motion, timestamps, mirrored, root)

        history_flat = (timestamps.unsqueeze(-1) + self.HistoryOffsets.to(timestamps.device)).flatten()
        history_roots = Tensor.Inverse(motion.GetModule(RootModule).GetTransforms(history_flat, mirrored=mirrored))
        history_frames_flat = self.ExtractFrameFeatures(motion, history_flat, mirrored, history_roots)

        x_history_flattened = history_frames_flat.reshape(len(timestamps), WINDOW_SIZE * FRAME_DIM)

        style_label = 0 
        name = motion.Name.lower()
        if "angry" in name: style_label = 0
        elif "depressed" in name: style_label = 1
        elif "drunk" in name: style_label = 2

        condition = torch.zeros(len(timestamps), NUM_STYLES)
        condition[:, style_label] = 1.0
        return (x_history_flattened, frames, Tensor.ToDevice(condition))

def main():
    AI4Animation(Program(), mode=AI4Animation.Mode.HEADLESS)

if __name__ == "__main__":
    main()

# # douleuei me to new_flow matcing alla xalia
# # Copyright (c) Meta Platforms, Inc. and affiliates.
# import os
# import sys
# from pathlib import Path
# import numpy as np

# import torch
# import matplotlib.pyplot as plt
# from ai4animation import (
#     AI4Animation, DataSampler, Dataset, FeedTensor, MirrorModule, 
#     MotionModule, RootModule, Tensor, Transform, Utility, Vector3
# )
# from ai4animation.AI.Models.AutoencoderVAE import VAEAutoencoder

# # ΕΙΣΑΓΩΓΗ ΤΟΥ ΔΙΚΟΥ ΣΟΥ FLOW MATCHING ΜΟΝΤΕΛΟΥ (Κλάση: Model)
# from ai4animation.AI.Models.New_FlowMatching import Model as FlowMatchingModel 

# SCRIPT_DIR = Path(__file__).parent
# ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/100style")
# sys.path.insert(0, ASSETS_PATH) 
# import Definitions

# EPOCH_COUNT = 150
# BATCH_SIZE = 32
# FRAMERATE = 30
# BONES = Definitions.FULL_BODY_NAMES
# FRAME_DIM = 12 * len(BONES)
# LATENT_DIM = 256

# WINDOW_SIZE = 15 
# NUM_STYLES = 3   # 3 Συναισθήματα: Angry, Depressed, Drunk
# CONDITION_MULTIPLIER = 10 

# class Program:
#     def Start(self):
#         Utility.SetSeed(23456)

#         self.Dataset = Dataset(
#             os.path.join(ASSETS_PATH, "Motions"),
#             [
#                 lambda x: RootModule(x, Definitions.HipName, Definitions.LeftHipName, Definitions.RightHipName, Definitions.LeftShoulderName, Definitions.RightShoulderName, Definitions.NeckName),
#                 lambda x: MotionModule(x),
#                 lambda x: MirrorModule(x, Vector3.Axis.ZPositive, Vector3.Create(0, 0, 180)),
#             ],
#         )

#         self.DataSampler = DataSampler(
#             self.Dataset,
#             framerate=FRAMERATE,
#             batch_size=BATCH_SIZE,
#             function=self.GetTrainingFeatures,
#         )

#         print("Loading pre-trained VAE...")
#         if os.path.exists("vae_full_model.pth"):
#             self.VAE = torch.load("vae_full_model_100style.pth", weights_only=False)
#             self.VAE = Tensor.ToDevice(self.VAE)
#         else:
#             raise Exception("Δεν βρέθηκε το vae_full_model.pth! Τρέξε πρώτα το ProgramVAE.py.")
            
#         self.VAE.eval()
#         for param in self.VAE.parameters():
#             param.requires_grad = False

#         # --- ΑΡΧΙΚΟΠΟΙΗΣΗ ΤΟΥ ΔΙΚΟΥ ΣΟΥ FLOW MATCHING ΜΟΝΤΕΛΟΥ ---
#         condition_dim = (WINDOW_SIZE * LATENT_DIM) + (NUM_STYLES * CONDITION_MULTIPLIER)
        
#         self.Network = Tensor.ToDevice(
#             FlowMatchingModel(
#                 cond_dim=condition_dim,
#                 target_dim=LATENT_DIM,
#                 hidden_dim=512,
#                 dropout=0.1,
#                 steps=10
#             )
#         )

#         self.HistoryOffsets = torch.arange(-WINDOW_SIZE, 0) / FRAMERATE
#         self.Trainer = self.Training()

#     def Standalone(self):
#         pass

#     def Update(self):
#         try:
#             next(self.Trainer)
#         except StopIteration:
#             pass

#     def EncodeBatch(self, raw_data, window_size):
#         batch_size = raw_data.shape[0]
#         raw_reshaped = raw_data.view(-1, FRAME_DIM)
#         norm_raw = self.VAE.Statistics.Normalize(Tensor.ToDevice(raw_reshaped.clone().detach().float()))
#         hidden = self.VAE.EncoderBody(norm_raw)
#         mu = self.VAE.fc_mu(hidden)
#         return mu.view(batch_size, window_size * LATENT_DIM)

#     def Training(self):
#         print("Splitting dataset into Training (80%) and Validation (20%)...")
#         all_batches = list(self.DataSampler.SampleBatchesWithinMotions(1, EPOCH_COUNT))
        
#         split_idx = int(0.8 * len(all_batches))
#         train_batches, val_batches = all_batches[:split_idx], all_batches[split_idx:]
#         total_train_samples = sum([b[1].shape[0] for b in train_batches])
        
#         self.Optimizer = Utility.CosineAnnealingOptimizer(
#             self.Network.parameters(), self.DataSampler.BatchSize, total_train_samples
#         )
        
#         train_losses, val_losses = [], []
#         best_val_loss = float('inf')

#         for epoch in range(1, EPOCH_COUNT + 1):
#             print(f"\n--- Epoch {epoch}/{EPOCH_COUNT} ---")
            
#             # --- ΕΚΠΑΙΔΕΥΣΗ FLOW MATCHING ---
#             self.Network.train()
#             epoch_train_loss = 0.0
            
#             for i, batch_data in enumerate(train_batches):
#                 xBatch_raw, yBatch_raw, condition = batch_data
                
#                 with torch.no_grad():
#                     xBatch_latent = self.EncodeBatch(xBatch_raw, WINDOW_SIZE)
#                     x_1 = self.EncodeBatch(yBatch_raw, 1) 

#                 # History Dropout
#                 if torch.rand(1).item() < (0.30 * (epoch / EPOCH_COUNT)):
#                     xBatch_latent = xBatch_latent * 0.0

#                 condition_amplified = condition.repeat(1, CONDITION_MULTIPLIER)
#                 cond_input = torch.cat([xBatch_latent, condition_amplified], dim=1)

#                 # Η ΜΑΓΕΙΑ ΕΔΩ: Η κλάση σου υπολογίζει το Loss μόνη της!
#                 _, loss_dict = self.Network.learn(cond_input, x_1, update_statistics=(epoch == 1))
                
#                 if isinstance(loss_dict, dict):
#                     loss = sum(loss_dict.values())
#                 else:
#                     loss = loss_dict
                
#                 self.Optimizer.Update(x_1.shape[0], loss)
#                 epoch_train_loss += loss.item()
                
#                 print(f"Training Progress: {100 * (i + 1) / len(train_batches):.1f}%", end="\r")
#                 yield
                
#             print(" " * 50, end="\r")
#             avg_train_loss = epoch_train_loss / len(train_batches)
#             train_losses.append(avg_train_loss)

#             # --- VALIDATION ---
#             self.Network.eval()
#             epoch_val_loss = 0.0
#             with torch.no_grad():
#                 for i, batch_data in enumerate(val_batches):
#                     xBatch_raw, yBatch_raw, condition = batch_data
#                     xBatch_latent = self.EncodeBatch(xBatch_raw, WINDOW_SIZE)
#                     x_1 = self.EncodeBatch(yBatch_raw, 1)
                    
#                     cond_input = torch.cat([xBatch_latent, condition.repeat(1, CONDITION_MULTIPLIER)], dim=1)
                    
#                     # Ζητάμε το Loss χωρίς να ανανεώνουμε στατιστικά
#                     _, loss_dict = self.Network.learn(cond_input, x_1, update_statistics=False)
#                     loss = sum(loss_dict.values()).item() if isinstance(loss_dict, dict) else loss_dict.item()
                    
#                     epoch_val_loss += loss
#                     yield
                    
#             avg_val_loss = epoch_val_loss / len(val_batches)
#             val_losses.append(avg_val_loss)
            
#             print(f"Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
            
#             if avg_val_loss < best_val_loss:
#                 best_val_loss = avg_val_loss
#                 print(">>> New best validation loss! <<<")

#             self.PlotTrainVal(train_losses, val_losses, epoch)

#         plt.ioff()
#         plt.savefig("loss_history_FlowMatching.png", dpi=300, bbox_inches='tight')
#         torch.save(self.Network, "100style_bonus_flow_matching_full.pth")
#         print("\n>>> Το Flow Matching μοντέλο αποθηκεύτηκε επιτυχώς! <<<")
#         plt.show()

#     def PlotTrainVal(self, train_losses, val_losses, epoch):
#         plt.ion()
#         plt.clf()
#         plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')
#         plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='orange')
#         plt.title(f'Flow Matching Vector Field Loss ({epoch}/{EPOCH_COUNT})')
#         plt.yscale('log')
#         plt.legend()
#         plt.grid(True, alpha=0.5)
#         plt.pause(0.01)

#     def ExtractFrameFeatures(self, motion, timestamps, mirrored, root):
#         transforms = Transform.TransformationFrom(motion.GetBoneTransformations(timestamps, BONES, mirrored=mirrored), root.reshape(-1, 1, 4, 4))
#         velocities = Vector3.DirectionFrom(motion.GetBoneVelocities(timestamps, BONES, mirrored=mirrored), root.reshape(-1, 1, 4, 4))
#         inputs = FeedTensor("Frame", (len(timestamps), FRAME_DIM))
#         inputs.Feed(Transform.GetPosition(transforms))
#         inputs.Feed(Transform.GetAxisZ(transforms))
#         inputs.Feed(Transform.GetAxisY(transforms))
#         inputs.Feed(velocities)
#         return inputs.GetTensor()

#     def GetTrainingFeatures(self, batch):
#         motion, timestamps = batch
#         if isinstance(timestamps, np.ndarray): timestamps = torch.from_numpy(timestamps)
#         mirrored = Tensor.RandomBool()

#         root = Tensor.Inverse(motion.GetModule(RootModule).GetTransforms(timestamps, mirrored=mirrored))
#         frames = self.ExtractFrameFeatures(motion, timestamps, mirrored, root)

#         history_flat = (timestamps.unsqueeze(-1) + self.HistoryOffsets.to(timestamps.device)).flatten()
#         history_roots = Tensor.Inverse(motion.GetModule(RootModule).GetTransforms(history_flat, mirrored=mirrored))
#         history_frames_flat = self.ExtractFrameFeatures(motion, history_flat, mirrored, history_roots)

#         x_history_flattened = history_frames_flat.reshape(len(timestamps), WINDOW_SIZE * FRAME_DIM)

#         style_label = 0 
#         name = motion.Name.lower()
#         if "angry" in name: style_label = 0
#         elif "depressed" in name: style_label = 1
#         elif "drunk" in name: style_label = 2

#         condition = torch.zeros(len(timestamps), NUM_STYLES)
#         condition[:, style_label] = 1.0
#         return (x_history_flattened, frames, Tensor.ToDevice(condition))

# def main():
#     AI4Animation(Program(), mode=AI4Animation.Mode.HEADLESS)

# if __name__ == "__main__":
#     main()