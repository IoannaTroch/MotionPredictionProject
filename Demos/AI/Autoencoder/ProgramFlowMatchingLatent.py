# ============================================================================
# Erwthma 5 (meros B): PROBABILISTIC autoregressive montelo me FLOW MATCHING
# ston LATENT xwro tou VAE (h pithanotiki ekdosi tou Erwthmatos 4).
#
# Roh:
#   1) fortwnoume ton ekpaideumeno VAE (vae_full_model.pth apo to Erwthma 2)
#      kai ton pagwnoume (frozen).
#   2) kanoume encode ta prohgoumena kare KAI to target kare se latent (mu).
#   3) to Flow Matching mathainei na PARAGEI to epomeno LATENT dianysma me vash
#      ena parathuro apo prohgoumena latent dianysmata (Latent Generation).
#   4) gia thn teliki kinhsh apokwdikopoioume to provlepomeno latent me ton VAE.
#
# Otan teleiwsei to training apothikeuei: flow_matching_latent_model.pth
# Gia na to deis sto GUI trekse meta to: ProgramFlowMatchingLatentInference.py
# ============================================================================
import os
import sys
from pathlib import Path
import numpy as np

import torch
import matplotlib.pyplot as plt
from ai4animation import (
    Actor,
    AI4Animation,
    DataSampler,
    Dataset,
    FeedTensor,
    MirrorModule,
    MotionEditor,
    MotionModule,
    ReadTensor,
    RootModule,
    Rotation,
    Tensor,
    Transform,
    Utility,
    Vector3,
)
from ai4animation.AI.Models import FlowMatching
from ai4animation.AI.Models.AutoencoderVAE import VAEAutoencoder

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/Cranberry")

sys.path.append(ASSETS_PATH)
import Definitions

EPOCH_COUNT = 150
BATCH_SIZE = 32
FRAMERATE = 30
DRAW_INTERVAL = 500
BONES = Definitions.FULL_BODY_NAMES
FRAME_DIM = 12 * len(BONES)      # 600 raw diastaseis
LATENT_DIM = 256                 # 256 latent apo ton VAE
HIDDEN_DIM = 512
WINDOW_SIZE = 5
FLOW_STEPS = 10
VAE_PATH = "vae_full_model.pth"
MODEL_PATH = "flow_matching_latent_model.pth"


class Program:
    def Start(self):
        Utility.SetSeed(23456)

        self.Dataset = Dataset(
            os.path.join(ASSETS_PATH, "Motions"),
            [
                lambda x: RootModule(
                    x,
                    Definitions.HipName,
                    Definitions.LeftHipName,
                    Definitions.RightHipName,
                    Definitions.LeftShoulderName,
                    Definitions.RightShoulderName,
                    Definitions.NeckName,
                ),
                lambda x: MotionModule(x),
                lambda x: MirrorModule(
                    x, Vector3.Axis.ZPositive, Vector3.Create(0, 0, 180)
                ),
            ],
        )

        self.DataSampler = DataSampler(
            self.Dataset,
            framerate=FRAMERATE,
            batch_size=BATCH_SIZE,
            function=self.GetTrainingFeatures,
        )

        # ---- fortwsh tou pre-trained VAE (Erwthma 2) ----
        print("Loading pre-trained VAE...")
        if os.path.exists(VAE_PATH):
            self.VAE = torch.load(VAE_PATH, weights_only=False)
            self.VAE = Tensor.ToDevice(self.VAE)
            print("VAE full model loaded successfully!")
        else:
            print(f"WARNING: '{VAE_PATH}' not found! Run ProgramVAE.py first.")
            self.VAE = Tensor.ToDevice(VAEAutoencoder(feature_dim=FRAME_DIM, latent_dim=LATENT_DIM))

        self.VAE.eval()
        for param in self.VAE.parameters():
            param.requires_grad = False

        # ---- Flow Matching ston LATENT xwro ----
        #   condition = window * latent_dim,  target = latent_dim
        self.Network = Tensor.ToDevice(
            FlowMatching.Model(
                cond_dim=WINDOW_SIZE * LATENT_DIM,
                target_dim=LATENT_DIM,
                hidden_dim=HIDDEN_DIM,
                steps=FLOW_STEPS,
            )
        )

        self.HistoryOffsets = torch.arange(-WINDOW_SIZE, 0) / FRAMERATE
        self.EditorHistory = torch.zeros(1, WINDOW_SIZE, LATENT_DIM)  # to istoriko einai latent

        self.Trainer = self.Training()

    def Standalone(self):
        entity = AI4Animation.Scene.AddEntity("Trainer")
        self.Editor = entity.AddComponent(
            MotionEditor,
            self.Dataset,
            os.path.join(ASSETS_PATH, "Model.glb"),
            BONES,
        )
        self.Actor = AI4Animation.Scene.AddEntity("Actor").AddComponent(
            Actor, os.path.join(ASSETS_PATH, "Model.glb"), BONES
        )
        self.Actor.SkinnedMesh.SetColor(AI4Animation.Color.RED)
        AI4Animation.Standalone.Camera.SetTarget(self.Actor.Entity)

    def Update(self):
        try:
            next(self.Trainer)
        except StopIteration:
            pass

    # metatrepei raw kare -> latent (xrhsimopoiei mono to mu, opws sto validation tou VAE)
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
            total_train_samples,
        )

        print(f"Total batches: {total_batches} | Train: {len(train_batches)} | Val: {len(val_batches)}")

        train_losses_history = []
        val_losses_history = []
        best_val_loss = float("inf")

        for epoch in range(1, EPOCH_COUNT + 1):
            print(f"\n--- Epoch {epoch}/{EPOCH_COUNT} ---")

            # -------- TRAINING --------
            self.Network.train()
            epoch_train_loss = 0.0
            for i, batch_data in enumerate(train_batches):
                xBatch_raw = batch_data[0]  # parelthon (raw window)
                yBatch_raw = batch_data[1]  # mellon (raw, ena kare)

                with torch.no_grad():
                    xBatch_latent = self.EncodeBatch(xBatch_raw, WINDOW_SIZE)
                    yBatch_latent = self.EncodeBatch(yBatch_raw, 1)

                _, loss = self.Network.learn(xBatch_latent, yBatch_latent, epoch == 1)
                tensor_loss = sum(loss.values()) if isinstance(loss, dict) else loss

                self.Optimizer.Update(yBatch_raw.shape[0], tensor_loss)
                epoch_train_loss += tensor_loss.item()

                progress = 100 * (i + 1) / len(train_batches)
                print(f"Training Progress: {progress:.1f}%", end="\r")
                yield
            print(" " * 50, end="\r")

            avg_train_loss = epoch_train_loss / len(train_batches)
            train_losses_history.append(avg_train_loss)

            # -------- VALIDATION --------
            self.Network.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for i, batch_data in enumerate(val_batches):
                    xBatch_raw = batch_data[0]
                    yBatch_raw = batch_data[1]
                    xBatch_latent = self.EncodeBatch(xBatch_raw, WINDOW_SIZE)
                    yBatch_latent = self.EncodeBatch(yBatch_raw, 1)

                    _, loss = self.Network.learn(xBatch_latent, yBatch_latent, False)
                    tensor_loss = sum(loss.values()) if isinstance(loss, dict) else loss
                    epoch_val_loss += tensor_loss.item()

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
        plt.savefig("loss_history_FlowMatchingLatent.png", dpi=300, bbox_inches="tight")

        torch.save(self.Network, MODEL_PATH)
        print(f"The model was saved successfully to '{MODEL_PATH}'!")
        plt.show()

    def PlotTrainVal(self, train_losses, val_losses, epoch):
        plt.ion()
        plt.clf()
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss", color="blue", linewidth=2)
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", color="orange", linewidth=2)
        plt.title(f"Flow Matching (Latent) Loss (Epoch {epoch}/{EPOCH_COUNT})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (Log Scale)")
        plt.yscale("log")
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

        return (x_history_flattened, frames)

    def GetEditorFeatures(self):
        features = FeedTensor("X", FRAME_DIM)
        root = self.Editor.Actor.Root
        transforms = Transform.TransformationTo(self.Editor.Actor.GetTransforms(BONES), root)
        velocities = Vector3.DirectionTo(self.Editor.Actor.GetVelocities(BONES), root)
        features.Feed(Transform.GetPosition(transforms))
        features.Feed(Transform.GetAxisZ(transforms))
        features.Feed(Transform.GetAxisY(transforms))
        features.Feed(velocities)
        return features.GetTensor()

    def Draw(self):
        self.Network.eval()
        with torch.no_grad():
            current_frame_raw = self.GetEditorFeatures().unsqueeze(0)
            current_latent = self.EncodeBatch(current_frame_raw, 1).unsqueeze(1)  # [1,1,LATENT_DIM]

            self.EditorHistory = torch.cat([self.EditorHistory[:, 1:, :], current_latent], dim=1)
            cond = self.EditorHistory.reshape(1, WINDOW_SIZE * LATENT_DIM)

            # Flow Matching paragei to epomeno latent
            predicted_latent = self.Network(cond, steps=FLOW_STEPS)

            # o VAE apokwdikopoiei to provlepomeno latent se raw kinhsh
            predicted_raw_norm = self.VAE.Decoder(predicted_latent)
            predicted_raw = self.VAE.Statistics.Denormalize(predicted_raw_norm)

            yPred = Tensor.ToNumPy(predicted_raw)
            output = ReadTensor("Y", yPred)

            self.Actor.Root = self.Editor.Actor.Root
            self.Actor.SetPositions(Vector3.PositionFrom(output.ReadVector3(len(BONES)), self.Actor.Root))
            self.Actor.SetRotations(Rotation.RotationFrom(output.ReadRotation3D(len(BONES)), self.Actor.Root))
            self.Actor.SetVelocities(Vector3.DirectionFrom(output.ReadVector3(len(BONES)), self.Actor.Root))
            for bone in self.Actor.Bones:
                bone.RestoreLength()
            self.Actor.RestoreBoneAlignments()
            self.Actor.SyncToScene()
        self.Network.train()


def main():
    AI4Animation(Program(), mode=AI4Animation.Mode.HEADLESS)


if __name__ == "__main__":
    main()
