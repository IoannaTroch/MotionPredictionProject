# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
from pathlib import Path
import numpy as np

import torch
from ai4animation import (
    Actor,
    AI4Animation,
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

# Import the CNN-MLP model — place CNNMLP.py in ai4animation/AI/Models/
from ai4animation.AI.Models import CNNMLP

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent / "_ASSETS_/Cranberry")

sys.path.append(ASSETS_PATH)
import Definitions

EPOCH_COUNT   = 100
BATCH_SIZE    = 32
FRAMERATE     = 30
DRAW_INTERVAL = 500
BONES         = Definitions.FULL_BODY_NAMES

# Each joint: position (3) + forward (3) + upward (3) + velocity (3) = 12
FRAME_DIM     = 12 * len(BONES)

# Model hyper-parameters
WINDOW_SIZE   = 5    # past frames to look back (same as plain MLP)
CNN_CHANNELS  = 128  # feature maps per conv layer (lighter)
HIDDEN_DIM    = 512  # MLP hidden layer size (same as original MLP)


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

        self.Network = Tensor.ToDevice(
            CNNMLP.Model(
                frame_dim=FRAME_DIM,
                window_size=WINDOW_SIZE,
                cnn_channels=CNN_CHANNELS,
                hidden_dim=HIDDEN_DIM,
            )
        )

        self.Optimizer = Utility.CosineAnnealingOptimizer(
            self.Network.parameters(),
            self.DataSampler.BatchSize,
            self.DataSampler.SampleCount,
        )

        self.LossHistory = Plotting.LossHistory(
            "Loss History", drawInterval=DRAW_INTERVAL, yScale="log"
        )

        # Past frame offsets: [-5/30s, -4/30s, ..., -1/30s]
        self.HistoryOffsets = torch.arange(-WINDOW_SIZE, 0) / FRAMERATE

        # Rolling inference buffer: [1, WindowSize, FrameDim], starts on CPU
        self.EditorHistory = torch.zeros(1, WINDOW_SIZE, FRAME_DIM)

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

    def Training(self):
        for epoch in range(1, EPOCH_COUNT + 1):
            print("Epoch", epoch)
            for batch_data in self.DataSampler.SampleBatchesWithinMotions(
                epoch, EPOCH_COUNT
            ):
                x_windows = batch_data[0]   # [Batch, WindowSize, FrameDim]
                y_frames  = batch_data[1]   # [Batch, FrameDim]

                loss = self.Network.learn(x_windows, y_frames, epoch == 1)
                self.Optimizer.Update(y_frames.shape[0], loss["MSE Loss"])

                for k, v in loss.items():
                    self.LossHistory.Add((Plotting.ToNumpy(v), k))

                yield
            self.LossHistory.Print()

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------
    def ExtractFrameFeatures(self, motion, timestamps, mirrored, root):
        """Returns [N, FrameDim] bone features relative to the root."""
        transforms = Transform.TransformationFrom(
            motion.GetBoneTransformations(timestamps, BONES, mirrored=mirrored),
            root.reshape(-1, 1, 4, 4),
        )
        velocities = Vector3.DirectionFrom(
            motion.GetBoneVelocities(timestamps, BONES, mirrored=mirrored),
            root.reshape(-1, 1, 4, 4),
        )
        feed = FeedTensor("Frame", (len(timestamps), FRAME_DIM))
        feed.Feed(Transform.GetPosition(transforms))
        feed.Feed(Transform.GetAxisZ(transforms))
        feed.Feed(Transform.GetAxisY(transforms))
        feed.Feed(velocities)
        return feed.GetTensor()

    def GetTrainingFeatures(self, batch):
        """
        Returns (x_windows, y_frames):
          x_windows : [Batch, WindowSize, FrameDim]  — past context, kept 3-D for CNN
          y_frames  : [Batch, FrameDim]              — ground-truth next frame
        """
        motion, timestamps = batch

        if isinstance(timestamps, np.ndarray):
            timestamps = torch.from_numpy(timestamps)

        mirrored = Tensor.RandomBool()

        root = Tensor.Inverse(
            motion.GetModule(RootModule).GetTransforms(timestamps, mirrored=mirrored)
        )

        # Target frames
        y_frames = self.ExtractFrameFeatures(motion, timestamps, mirrored, root)

        # History frames: WINDOW_SIZE frames before each timestamp
        history_timestamps = timestamps.unsqueeze(-1) + self.HistoryOffsets.to(
            timestamps.device
        )
        history_flat = history_timestamps.flatten()

        history_roots = Tensor.Inverse(
            motion.GetModule(RootModule).GetTransforms(history_flat, mirrored=mirrored)
        )
        history_features = self.ExtractFrameFeatures(
            motion, history_flat, mirrored, history_roots
        )

        # Keep as [Batch, WindowSize, FrameDim] — do NOT flatten for the CNN
        x_windows = history_features.reshape(len(timestamps), WINDOW_SIZE, FRAME_DIM)

        return (x_windows, y_frames)

    def GetEditorFeatures(self):
        """Returns a [1, FrameDim] tensor from the live editor actor."""
        feed = FeedTensor("X", FRAME_DIM)
        root = self.Editor.Actor.Root
        transforms = Transform.TransformationTo(
            self.Editor.Actor.GetTransforms(BONES), root
        )
        velocities = Vector3.DirectionTo(
            self.Editor.Actor.GetVelocities(BONES), root
        )
        feed.Feed(Transform.GetPosition(transforms))
        feed.Feed(Transform.GetAxisZ(transforms))
        feed.Feed(Transform.GetAxisY(transforms))
        feed.Feed(velocities)
        return feed.GetTensor()

    def Draw(self):
        self.Network.eval()
        with torch.no_grad():
            # current_frame: [1, 1, FrameDim]
            current_frame = self.GetEditorFeatures().unsqueeze(0).unsqueeze(1)

            # Move buffer to same device as current_frame (lazy, first call only)
            self.EditorHistory = self.EditorHistory.to(current_frame.device)

            # Slide the rolling window forward
            self.EditorHistory = torch.cat(
                [self.EditorHistory[:, 1:, :], current_frame], dim=1
            )

            # Predict one frame; squeeze [1, 1, FrameDim] -> [1, FrameDim]
            raw = self.Network(self.EditorHistory, generate_steps=1)
            yPred = Tensor.ToNumPy(raw.squeeze(1))

            output = ReadTensor("Y", yPred)

            self.Actor.Root = self.Editor.Actor.Root
            self.Actor.SetPositions(
                Vector3.PositionFrom(output.ReadVector3(len(BONES)), self.Actor.Root)
            )
            self.Actor.SetRotations(
                Rotation.RotationFrom(
                    output.ReadRotation3D(len(BONES)), self.Actor.Root
                )
            )
            self.Actor.SetVelocities(
                Vector3.DirectionFrom(output.ReadVector3(len(BONES)), self.Actor.Root)
            )
            for bone in self.Actor.Bones:
                bone.RestoreLength()
            self.Actor.RestoreBoneAlignments()
            self.Actor.SyncToScene()

        self.Network.train()


def main():
    AI4Animation(Program(), mode=AI4Animation.Mode.HEADLESS)


if __name__ == "__main__":
    main()
