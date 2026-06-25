# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
from pathlib import Path
import numpy as np

import torch
import matplotlib.pyplot as plt
from ai4animation import (
    Actor, AI4Animation, LongShortTermMemory, DataSampler, Dataset,
    FeedTensor, MirrorModule, MotionEditor, MotionModule, Plotting,
    ReadTensor, RootModule, Rotation, Tensor, Transform, Utility, Vector3
)

from ai4animation.AI.Models.AutoencoderVAE import VAEAutoencoder

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/Cranberry")
sys.path.append(ASSETS_PATH)
import Definitions

EPOCH_COUNT = 300 
BATCH_SIZE = 32
FRAMERATE = 30
DRAW_INTERVAL = 500
BONES = Definitions.FULL_BODY_NAMES
FRAME_DIM = 12 * len(BONES)
LATENT_DIM = 256
HIDDEN_DIM = 512

WINDOW_SIZE = 15 
NUM_STYLES = 5   
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
        if os.path.exists("vae_full_model.pth"):
            self.VAE = torch.load("vae_full_model.pth", weights_only=False)
            self.VAE = Tensor.ToDevice(self.VAE)
            print("VAE full model loaded successfully!")
        else:
            print("WARNING: 'vae_full_model.pth' not found! Run ProgramVAE.py first.")
            self.VAE = Tensor.ToDevice(VAEAutoencoder(feature_dim=FRAME_DIM, latent_dim=LATENT_DIM))
            
        self.VAE.eval()
        for param in self.VAE.parameters():
            param.requires_grad = False

        self.Network = Tensor.ToDevice(
            LongShortTermMemory.Model(
                input_dim=(WINDOW_SIZE * LATENT_DIM) + (NUM_STYLES * CONDITION_MULTIPLIER),
                output_dim=LATENT_DIM,
                hidden_dim=HIDDEN_DIM,
                future_steps=1,
                num_layers=2
            )
        )

        self.HistoryOffsets = torch.arange(-WINDOW_SIZE, 0) / FRAMERATE
        self.Trainer = self.Training()

    def Standalone(self):
        pass 

    def Update(self):
        try:
            next(self.Trainer)
        except StopIteration as e:
            pass

    def EncodeBatch(self, raw_data, window_size):
        batch_size = raw_data.shape[0]
        raw_reshaped = raw_data.view(-1, FRAME_DIM)
        raw_tensor = Tensor.ToDevice(raw_reshaped.clone().detach().float())
        norm_raw = self.VAE.Statistics.Normalize(raw_tensor)
        hidden = self.VAE.EncoderBody(norm_raw)
        mu = self.VAE.fc_mu(hidden)
        return mu.view(batch_size, window_size * LATENT_DIM)

    def Training(self):
        print("Splitting dataset into Training (80%) and Validation (20%)...")
        all_batches = list(self.DataSampler.SampleBatchesWithinMotions(1, EPOCH_COUNT))
        
        total_batches = len(all_batches)
        split_idx = int(0.8 * total_batches)
        
        train_batches = all_batches[:split_idx]
        val_batches = all_batches[split_idx:]
        
        total_train_samples = sum([batch[1].shape[0] for batch in train_batches])
        
        self.Optimizer = Utility.CosineAnnealingOptimizer(
            self.Network.parameters(),
            self.DataSampler.BatchSize,
            total_train_samples
        )
        
        train_losses_history = []
        val_losses_history = []
        best_val_loss = float('inf')

        for epoch in range(1, EPOCH_COUNT + 1):
            print(f"\n--- Epoch {epoch}/{EPOCH_COUNT} ---")
            
            self.Network.train()
            epoch_train_loss = 0.0
            
            for i, batch_data in enumerate(train_batches):
                xBatch_raw = batch_data[0]
                yBatch_raw = batch_data[1]
                condition = batch_data[2]
                
                with torch.no_grad():
                    xBatch_latent = self.EncodeBatch(xBatch_raw, WINDOW_SIZE)
                    yBatch_latent = self.EncodeBatch(yBatch_raw, 1)

                current_dropout_prob = 0.30 * (epoch / EPOCH_COUNT)
                if torch.rand(1).item() < current_dropout_prob:
                    xBatch_latent = xBatch_latent * 0.0

                condition_amplified = condition.repeat(1, CONDITION_MULTIPLIER)
                xBatch_conditional = torch.cat([xBatch_latent, condition_amplified], dim=1)

                _, loss = self.Network.learn(xBatch_conditional, yBatch_latent, epoch == 1)
                
                if isinstance(loss, dict):
                    tensor_loss = sum(loss.values())
                else:
                    tensor_loss = loss
                    
                self.Optimizer.Update(yBatch_raw.shape[0], tensor_loss)
                epoch_train_loss += tensor_loss.item()
                
                progress = 100 * (i + 1) / len(train_batches)
                print(f"Training Progress: {progress:.1f}%", end="\r")
                yield
                
            print(" " * 50, end="\r")
            avg_train_loss = epoch_train_loss / len(train_batches)
            train_losses_history.append(avg_train_loss)

            self.Network.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for i, batch_data in enumerate(val_batches):
                    xBatch_raw = batch_data[0]
                    yBatch_raw = batch_data[1]
                    condition = batch_data[2]
                    
                    xBatch_latent = self.EncodeBatch(xBatch_raw, WINDOW_SIZE)
                    yBatch_latent = self.EncodeBatch(yBatch_raw, 1)
                    condition_amplified = condition.repeat(1, CONDITION_MULTIPLIER)
                    xBatch_conditional = torch.cat([xBatch_latent, condition_amplified], dim=1)
                    
                    input_norm = self.Network.InputStatistics.Normalize(xBatch_conditional)
                    output_norm = self.Network.OutputStatistics.Normalize(yBatch_latent)
                    
                    prediction = self.Network.Layers(input_norm)
                    loss = torch.nn.functional.mse_loss(prediction, output_norm).item()
                    
                    epoch_val_loss += loss
                    
                    progress = 100 * (i + 1) / len(val_batches)
                    print(f"Validation Progress: {progress:.1f}%", end="\r")
                    yield
            
            print(" " * 50, end="\r")
            avg_val_loss = epoch_val_loss / len(val_batches)
            val_losses_history.append(avg_val_loss)
            
            print(f"Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(">>> New best validation loss! <<<")

            self.PlotTrainVal(train_losses_history, val_losses_history, epoch)

        plt.ioff()
        plt.savefig("loss_history_ConditionalLSTM.png", dpi=300, bbox_inches='tight')
        
        save_path = os.path.join(SCRIPT_DIR, "conditional_lstm_full.pth")
        torch.save(self.Network, save_path)
        print(f"\n>>> Το πλήρες μοντέλο αποθηκεύτηκε επιτυχώς στο: {save_path} <<<")
        plt.show()

    def PlotTrainVal(self, train_losses, val_losses, epoch):
        plt.ion()
        plt.clf()
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue', linewidth=2)
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='orange', linewidth=2)
        plt.title(f'Conditional LSTM Loss (Epoch {epoch}/{EPOCH_COUNT})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Log Scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.pause(0.01)

    def ExtractFrameFeatures(self, motion, timestamps, mirrored, root):
        transforms = Transform.TransformationFrom(
            motion.GetBoneTransformations(timestamps, BONES, mirrored=mirrored),
            root.reshape(-1, 1, 4, 4),
        )
        velocities = Vector3.DirectionFrom(
            motion.GetBoneVelocities(timestamps, BONES, mirrored=mirrored),
            root.reshape(-1, 1, 4, 4),
        )
        inputs = FeedTensor("Frame", (len(timestamps), FRAME_DIM))
        inputs.Feed(Transform.GetPosition(transforms))
        inputs.Feed(Transform.GetAxisZ(transforms))
        inputs.Feed(Transform.GetAxisY(transforms))
        inputs.Feed(velocities)
        return inputs.GetTensor()

    def GetTrainingFeatures(self, batch):
        motion, timestamps = batch
        if isinstance(timestamps, np.ndarray):
            timestamps = torch.from_numpy(timestamps)
        mirrored = Tensor.RandomBool()

        root = Tensor.Inverse(motion.GetModule(RootModule).GetTransforms(timestamps, mirrored=mirrored))
        frames = self.ExtractFrameFeatures(motion, timestamps, mirrored, root)

        history_timestamps = timestamps.unsqueeze(-1) + self.HistoryOffsets.to(timestamps.device)
        history_flat = history_timestamps.flatten()

        history_roots = Tensor.Inverse(motion.GetModule(RootModule).GetTransforms(history_flat, mirrored=mirrored))
        history_frames_flat = self.ExtractFrameFeatures(motion, history_flat, mirrored, history_roots)

        x_history_windows = history_frames_flat.reshape(len(timestamps), WINDOW_SIZE, FRAME_DIM)
        x_history_flattened = x_history_windows.reshape(len(timestamps), WINDOW_SIZE * FRAME_DIM)

        style_label = 0 
        name = motion.Name.lower()
        
        if "standing" in name:
            style_label = 0
        elif "walking_forward" in name:
            style_label = 1
        elif "walking_backward" in name:
            style_label = 2
        elif "crouching" in name:
            style_label = 3
        elif "vr_beatsaber" in name:
            style_label = 4

        condition = torch.zeros(len(timestamps), NUM_STYLES)
        condition[:, style_label] = 1.0
        condition = Tensor.ToDevice(condition)

        return (x_history_flattened, frames, condition)

def main():
    AI4Animation(Program(), mode=AI4Animation.Mode.HEADLESS)

if __name__ == "__main__":
    main()