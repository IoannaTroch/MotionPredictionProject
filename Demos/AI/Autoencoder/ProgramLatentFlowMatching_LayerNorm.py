import os
import sys
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ai4animation import (
    Actor,
    AI4Animation,
    FlowMatching,
    DataSampler,
    Dataset,
    FeedTensor,
    MirrorModule,
    MotionEditor,
    MotionModule,
    Plotting,
    ReadTensor,
    RootModule,
    Rotation,
    Tensor,
    Transform,
    Utility,
    Vector3,
)

from ai4animation.AI.Models.AutoencoderLayerNorm import EnchancedAutoencoder

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/Cranberry")

sys.path.append(ASSETS_PATH)
import Definitions

EPOCH_COUNT = 150
BATCH_SIZE = 32
FRAMERATE = 30
DRAW_INTERVAL = 500
BONES = Definitions.FULL_BODY_NAMES
FRAME_DIM = 12 * len(BONES) 
LATENT_DIM = 256             
HIDDEN_DIM = 512
WINDOW_SIZE = 5              
TIME_EMB_DIM = 64
INTEGRATION_STEPS = 10    

AUTOENCODER_CKPT = "layernorm_full_model.pth"


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

        print("Loading pre-trained LayerNorm Autoencoder...")
        if os.path.exists(AUTOENCODER_CKPT):
            self.Autoencoder = torch.load(AUTOENCODER_CKPT, weights_only=False)
            self.Autoencoder = Tensor.ToDevice(self.Autoencoder)
            print(f"   loaded '{AUTOENCODER_CKPT}'")
        else:
            print(
                f"WARNING: '{AUTOENCODER_CKPT}' not found — running with an "
                f"untrained autoencoder. Run ProgramLayerNorm.py first."
            )
            self.Autoencoder = Tensor.ToDevice(
                EnchancedAutoencoder(feature_dim=FRAME_DIM, latent_dim=LATENT_DIM)
            )

        self.Autoencoder.eval()
        for p in self.Autoencoder.parameters():
            p.requires_grad = False

        self.Network = Tensor.ToDevice(
            FlowMatching.Model(
                cond_dim=WINDOW_SIZE * LATENT_DIM,
                target_dim=LATENT_DIM,
                hidden_dim=HIDDEN_DIM,
                steps=INTEGRATION_STEPS,
            )
        )

        self.HistoryOffsets = torch.arange(-WINDOW_SIZE, 0) / FRAMERATE

        self.EditorHistory = torch.zeros(1, WINDOW_SIZE, LATENT_DIM)

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

    def EncodeBatch(self, raw_data, window_size):
        batch_size = raw_data.shape[0]
        raw_reshaped = raw_data.view(-1, FRAME_DIM)
        raw_tensor = Tensor.ToDevice(raw_reshaped.clone().detach().float())
        norm_raw = self.Autoencoder.Statistics.Normalize(raw_tensor)
        latent = self.Autoencoder.Encoder(norm_raw)
        return latent.view(batch_size, window_size * LATENT_DIM)

    def Training(self):
        print("Splitting dataset into Training (80%) and Validation (20%)...")
        all_batches = list(self.DataSampler.SampleBatchesWithinMotions(1, EPOCH_COUNT))

        total_batches = len(all_batches)
        split_idx = int(0.8 * total_batches)
        train_batches = all_batches[:split_idx]
        val_batches = all_batches[split_idx:]

        total_train_samples = sum(b[1].shape[0] for b in train_batches)
        self.Optimizer = Utility.CosineAnnealingOptimizer(
            self.Network.parameters(),
            self.DataSampler.BatchSize,
            total_train_samples,
        )

        train_losses_history = []
        val_mse_history = []
        best_val_mse = float("inf")

        for epoch in range(1, EPOCH_COUNT + 1):
            print(f"\n--- Epoch {epoch}/{EPOCH_COUNT} ---")

            self.Network.train()
            epoch_train_loss = 0.0
            for i, batch_data in enumerate(train_batches):
                xBatch_raw, yBatch_raw = batch_data[0], batch_data[1]

                with torch.no_grad():
                    xBatch_latent = self.EncodeBatch(xBatch_raw, WINDOW_SIZE)
                    yBatch_latent = self.EncodeBatch(yBatch_raw, 1)

                _, loss = self.Network.learn(xBatch_latent, yBatch_latent, epoch == 1)
                fm_loss = sum(loss.values())

                self.Optimizer.Update(yBatch_raw.shape[0], fm_loss)
                epoch_train_loss += fm_loss.item()

                progress = 100 * (i + 1) / len(train_batches)
                print(f"Training Progress: {progress:.1f}%", end="\r")
                yield

            print(" " * 50, end="\r")
            avg_train_loss = epoch_train_loss / len(train_batches)
            train_losses_history.append(avg_train_loss)

            self.Network.eval()
            epoch_val_mse = 0.0
            with torch.no_grad():
                for i, batch_data in enumerate(val_batches):
                    xBatch_raw, yBatch_raw = batch_data[0], batch_data[1]
                    xBatch_latent = self.EncodeBatch(xBatch_raw, WINDOW_SIZE)
                    yBatch_latent = self.EncodeBatch(yBatch_raw, 1)

                    sampled_latent = self.Network.sample(xBatch_latent)

                    pred_n = self.Network.OutputStatistics.Normalize(sampled_latent)
                    targ_n = self.Network.OutputStatistics.Normalize(yBatch_latent)
                    mse = F.mse_loss(pred_n, targ_n).item()
                    epoch_val_mse += mse

                    progress = 100 * (i + 1) / len(val_batches)
                    print(f"Validation Progress: {progress:.1f}%", end="\r")
                    yield

            print(" " * 50, end="\r")
            avg_val_mse = epoch_val_mse / len(val_batches)
            val_mse_history.append(avg_val_mse)

            print(
                f"Train FM Loss: {avg_train_loss:.5f} | Val Sampled MSE: {avg_val_mse:.5f}"
            )

            if avg_val_mse < best_val_mse:
                best_val_mse = avg_val_mse
                print(">>> New best validation MSE! <<<")
                torch.save(self.Network.state_dict(), "fm_latent_best.pth")

            self.PlotTrainVal(train_losses_history, val_mse_history, epoch)

        plt.ioff()
        plt.savefig("loss_history_FM_latent.png", dpi=300, bbox_inches="tight")
        plt.show()

    def PlotTrainVal(self, train_losses, val_mse, epoch):
        plt.ion()
        plt.clf()
        plt.plot(
            range(1, len(train_losses) + 1),
            train_losses,
            label="Train FM Loss",
            color="blue",
            linewidth=2,
        )
        plt.plot(
            range(1, len(val_mse) + 1),
            val_mse,
            label="Val Sampled MSE (latent)",
            color="orange",
            linewidth=2,
        )
        plt.title(f"Latent Flow Matching (Epoch {epoch}/{EPOCH_COUNT})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (log scale)")
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

        root = Tensor.Inverse(
            motion.GetModule(RootModule).GetTransforms(timestamps, mirrored=mirrored)
        )

        frames = self.ExtractFrameFeatures(motion, timestamps, mirrored, root)

        history_timestamps = (
            timestamps.unsqueeze(-1) + self.HistoryOffsets.to(timestamps.device)
        )
        history_flat = history_timestamps.flatten()
        history_roots = Tensor.Inverse(
            motion.GetModule(RootModule).GetTransforms(history_flat, mirrored=mirrored)
        )
        history_frames_flat = self.ExtractFrameFeatures(
            motion, history_flat, mirrored, history_roots
        )
        x_history_windows = history_frames_flat.reshape(
            len(timestamps), WINDOW_SIZE, FRAME_DIM
        )
        x_history_flattened = x_history_windows.reshape(
            len(timestamps), WINDOW_SIZE * FRAME_DIM
        )

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
        with torch.no_grad():
            current_frame_raw = self.GetEditorFeatures().unsqueeze(0)
            current_latent = self.EncodeBatch(current_frame_raw, 1).unsqueeze(1)

            self.EditorHistory = torch.cat(
                [self.EditorHistory[:, 1:, :], current_latent], dim=1
            )
            history_flat = self.EditorHistory.reshape(1, WINDOW_SIZE * LATENT_DIM)

            predicted_latent = self.Network(history_flat)

            predicted_raw_norm = self.Autoencoder.Decoder(predicted_latent)
            predicted_raw = self.Autoencoder.Statistics.Denormalize(predicted_raw_norm)

            yPred = Tensor.ToNumPy(predicted_raw)
            output = ReadTensor("Y", yPred)

            self.Actor.Root = self.Editor.Actor.Root
            self.Actor.SetPositions(
                Vector3.PositionFrom(output.ReadVector3(len(BONES)), self.Actor.Root)
            )
            self.Actor.SetRotations(
                Rotation.RotationFrom(output.ReadRotation3D(len(BONES)), self.Actor.Root)
            )
            self.Actor.SetVelocities(
                Vector3.DirectionFrom(output.ReadVector3(len(BONES)), self.Actor.Root)
            )
            for bone in self.Actor.Bones:
                bone.RestoreLength()
            self.Actor.RestoreBoneAlignments()
            self.Actor.SyncToScene()


def main():
    AI4Animation(Program(), mode=AI4Animation.Mode.STANDALONE)


if __name__ == "__main__":
    main()
