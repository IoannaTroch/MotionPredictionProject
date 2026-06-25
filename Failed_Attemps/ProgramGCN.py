# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# GCN with Learnable Adjacency + Stride
# Training parameters identical to ProgramAutoregressionMLPstride.py
#
# Model: GCNLearnableAdj — skeleton-initialised adjacency matrix
#        made learnable so the model discovers joint relationships
#        beyond the physical skeleton structure.
#
# Stride: frames spaced STRIDE steps apart, same as MLPstride.
# LiveFrameBuffer: same ring buffer inference fix as MLPstride.

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

from ai4animation.AI.Models import GCNLearnableAdj

SCRIPT_DIR  = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/Cranberry")

sys.path.append(ASSETS_PATH)
import Definitions

# -----------------------------------------------------------------------
# Identical to ProgramAutoregressionMLPstride.py
# -----------------------------------------------------------------------
EPOCH_COUNT   = 150
BATCH_SIZE    = 32
FRAMERATE     = 30
DRAW_INTERVAL = 500
BONES         = Definitions.FULL_BODY_NAMES

NUM_JOINTS     = len(BONES)                   # 27
FEAT_PER_JOINT = 12                           # pos(3)+fwd(3)+up(3)+vel(3)
FRAME_DIM      = NUM_JOINTS * FEAT_PER_JOINT  # 324

WINDOW_SIZE   = 5
STRIDE        = 2

# -----------------------------------------------------------------------
# GCN-specific (kept minimal to match MLP complexity)
# -----------------------------------------------------------------------
HIDDEN_DIM     = 128
NUM_GCN_LAYERS = 4


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
            GCNLearnableAdj.Model(
                num_joints=NUM_JOINTS,
                feat_per_joint=FEAT_PER_JOINT,
                window_size=WINDOW_SIZE,
                hidden_dim=HIDDEN_DIM,
                num_gcn_layers=NUM_GCN_LAYERS,
            )
        )

        # Identical to ProgramAutoregressionMLPstride.py —
        # no lr or decay override, uses framework defaults (lr=1e-4, decay=1e-4)
        self.Optimizer = Utility.CosineAnnealingOptimizer(
            self.Network.parameters(),
            self.DataSampler.BatchSize,
            self.DataSampler.SampleCount,
        )

        self.LossHistory = Plotting.LossHistory(
            "Loss History", drawInterval=DRAW_INTERVAL, yScale="log"
        )

        # Strided history offsets — identical to MLPstride
        self.HistoryOffsets = (torch.arange(-WINDOW_SIZE, 0) * STRIDE) / FRAMERATE

        # Rolling inference buffer
        self.EditorHistory  = torch.zeros(1, WINDOW_SIZE, FRAME_DIM)

        # Ring buffer for live strided inference — identical to MLPstride
        self.LiveFrameBuffer = []

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
            for batch in self.DataSampler.SampleBatchesWithinMotions(epoch, EPOCH_COUNT):
                x_windows = batch[0]   # [Batch, WindowSize, FrameDim]
                y_frames  = batch[1]   # [Batch, FrameDim]

                loss = self.Network.learn(x_windows, y_frames, epoch == 1)
                self.Optimizer.Update(y_frames.shape[0], loss["MSE Loss"])

                for k, v in loss.items():
                    self.LossHistory.Add((Plotting.ToNumpy(v), k))

                yield
            self.LossHistory.Print()

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
        root     = Tensor.Inverse(
            motion.GetModule(RootModule).GetTransforms(timestamps, mirrored=mirrored)
        )

        y_frames = self.ExtractFrameFeatures(motion, timestamps, mirrored, root)

        # Strided history — identical to MLPstride
        history_timestamps = timestamps.unsqueeze(-1) + self.HistoryOffsets.to(timestamps.device)
        history_flat       = history_timestamps.flatten()
        history_roots      = Tensor.Inverse(
            motion.GetModule(RootModule).GetTransforms(history_flat, mirrored=mirrored)
        )
        history_features = self.ExtractFrameFeatures(
            motion, history_flat, mirrored, history_roots
        )

        # [Batch, WindowSize, FrameDim] — 3D for GCN, NOT flattened
        x_windows = history_features.reshape(len(timestamps), WINDOW_SIZE, FRAME_DIM)

        return (x_windows, y_frames)

    def GetEditorFeatures(self):
        features   = FeedTensor("X", FRAME_DIM)
        root       = self.Editor.Actor.Root
        transforms = Transform.TransformationTo(
            self.Editor.Actor.GetTransforms(BONES), root
        )
        velocities = Vector3.DirectionTo(
            self.Editor.Actor.GetVelocities(BONES), root
        )
        features.Feed(Transform.GetPosition(transforms))
        features.Feed(Transform.GetAxisZ(transforms))
        features.Feed(Transform.GetAxisY(transforms))
        features.Feed(velocities)
        return features.GetTensor()

    def Draw(self):
        self.Network.eval()
        with torch.no_grad():
            current_frame = self.GetEditorFeatures().unsqueeze(0).unsqueeze(1)

            # Ring buffer — identical to MLPstride
            self.LiveFrameBuffer.append(current_frame)
            if len(self.LiveFrameBuffer) > (WINDOW_SIZE * STRIDE):
                self.LiveFrameBuffer.pop(0)

            if len(self.LiveFrameBuffer) == (WINDOW_SIZE * STRIDE):
                strided_frames     = self.LiveFrameBuffer[::STRIDE]
                self.EditorHistory = torch.cat(strided_frames, dim=1)
                self.EditorHistory = self.EditorHistory.to(current_frame.device)
            else:
                self.EditorHistory = self.EditorHistory.to(current_frame.device)
                self.EditorHistory = torch.cat(
                    [self.EditorHistory[:, 1:, :], current_frame], dim=1
                )

            raw   = self.Network(self.EditorHistory, generate_steps=1)
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