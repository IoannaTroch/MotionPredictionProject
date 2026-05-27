# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
from pathlib import Path

import torch
from ai4animation import (
    Actor,
    AI4Animation,
    Autoencoder,
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

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/Cranberry")

sys.path.append(ASSETS_PATH)
import Definitions

EPOCH_COUNT = 150 
BATCH_SIZE = 32 # number of animation snapshots processed simultaneously
FRAMERATE = 30 # animation speed resolutions (frames per second)
DRAW_INTERVAL = 500 # how often to display metric plots
BONES = Definitions.FULL_BODY_NAMES # array os strings representing joints
FEATURE_DIM = 12 * len(BONES) # every joint has 12 dimensions
# position 3, forward 3, upward 3, velocity 3
HIDDEN_DIM = 512 # neurons in the hidden layers
LATENT_DIM = 256 # compressed latent space dimensions


class Program:
    def Start(self):
        Utility.SetSeed(23456)

        self.Dataset = Dataset( # creates the dataset based on motion files
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
                lambda x: MotionModule(x), # tracks joint rotations and translational tranformations frame-by-frame
                lambda x: MirrorModule( # flips values across z-axis to artificially double dataset volume
                    x, Vector3.Axis.ZPositive, Vector3.Create(0, 0, 180)
                ),
            ],
        )

        self.DataSampler = DataSampler( # mini-batch setup
            self.Dataset,
            framerate=FRAMERATE,
            batch_size=BATCH_SIZE,
            function=self.GetTrainingFeatures, # feeds data matrices through our tracking extractor
        )

        self.Network = Tensor.ToDevice(
            Autoencoder.Model(
                feature_dim=FEATURE_DIM,
                hidden_dim=HIDDEN_DIM,
                latent_dim=LATENT_DIM,
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
        except StopIteration as e:
            pass

    def Training(self):
        for epoch in range(1, EPOCH_COUNT + 1):
            print("Epoch", epoch)
            # fetch batched frame blocks randomly sliced across your motion folders
            for batch in self.DataSampler.SampleBatchesWithinMotions(
                epoch, EPOCH_COUNT
            ):
                _, loss = self.Network.learn(batch, epoch == 1) # forward and backward pass
                self.Optimizer.Update(batch.shape[0], loss["MSE Loss"])
                for k, v in loss.items():
                    self.LossHistory.Add((Plotting.ToNumpy(v), k))
                yield # pauses for syncing
            self.LossHistory.Print()

    def GetTrainingFeatures(self, batch):
        """Converts raw dataset vectors into space-relative neural network features"""
        motion, timestamps = batch
        mirrored = Tensor.RandomBool() # random mirroring for data augmentation

        # empty feature buffer
        inputs = FeedTensor("X", (len(timestamps), FEATURE_DIM))

        # calculates the inverse root transformation matrix (normalizes tracking space context)
        root = Tensor.Inverse(
            motion.GetModule(RootModule).GetTransforms(timestamps, mirrored=mirrored)
        )

        # Inputs
        # computers space-relative joint transformations
        transforms = Transform.TransformationFrom(
            motion.GetBoneTransformations(timestamps, BONES, mirrored=mirrored),
            root.reshape(-1, 1, 4, 4), # align dimensions for processing
        )
        # computer space-relative directional velocity vectors
        velocities = Vector3.DirectionFrom(
            motion.GetBoneVelocities(timestamps, BONES, mirrored=mirrored),
            root.reshape(-1, 1, 4, 4), 
        )
        # adds features to buffer
        inputs.Feed(Transform.GetPosition(transforms)) # local positions (x, y, z)
        inputs.Feed(Transform.GetAxisZ(transforms)) # forward vectors
        inputs.Feed(Transform.GetAxisY(transforms)) # upward vectors
        inputs.Feed(velocities) # dynamic linear velocity

        return inputs.GetTensor()

    def GetEditorFeatures(self):
        """Extracts tracking features from the interactive 3D editor character asset"""
        features = FeedTensor("X", FEATURE_DIM)
        root = self.Editor.Actor.Root
        # normalise 3D editor coords to match root orientation layout logic
        transforms = Transform.TransformationTo(
            self.Editor.Actor.GetTransforms(BONES), root
        )
        velocities = Vector3.DirectionTo(self.Editor.Actor.GetVelocities(BONES), root)
        features.Feed(Transform.GetPosition(transforms))
        features.Feed(Transform.GetAxisZ(transforms))
        features.Feed(Transform.GetAxisY(transforms))
        features.Feed(velocities)
        return features.GetTensor()

    def Draw(self):
        """Translates neural predictions back into skeletal positions inside the 3D engine"""
        self.Network.eval()
        with torch.no_grad():
            xBatch = self.GetEditorFeatures()
            yPred = Tensor.ToNumPy(self.Network(xBatch))
            output = ReadTensor("Y", yPred)
            self.Actor.Root = self.Editor.Actor.Root # Snap the prediction actor's core location to match the master trajectory root
            self.Actor.SetPositions( # Reposition character bone arrays based on parsed network output features
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
    AI4Animation(Program(), mode=AI4Animation.Mode.STANDALONE)


if __name__ == "__main__":
    main()
