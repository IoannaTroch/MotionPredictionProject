# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
from pathlib import Path
import numpy as np

import torch
from ai4animation import (
    AI4Animation,
    DataSampler,
    Dataset,
    FeedTensor,
    LongShortTermMemory,  # swapped: was MultiLayerPerceptron
    MirrorModule,
    MotionEditor,
    MotionModule,
    Plotting,
    ReadTensor,
    RootModule,
    Rotation,
    Tensor,
    TimeSeries,
    Transform,
    Utility,
    Vector3,
)

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/Cranberry")
sys.path.append(ASSETS_PATH)
import Definitions

EPOCH_COUNT = 150
BATCH_SIZE = 32
FRAMERATE = 30
DRAW_INTERVAL = 500
BONES = Definitions.FULL_BODY_NAMES
FUTURE_SAMPLES = 6
INPUT_DIM = 12 * len(BONES)
OUTPUT_DIM = FUTURE_SAMPLES * 4 + FUTURE_SAMPLES * len(BONES) * 9
HIDDEN_DIM = 1024
NUM_LAYERS = 2  # LSTM layers
STEP_OUTPUT_DIM = 4 + len(BONES) * 9  # 4 root values + (joints * 9 values)


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
            LongShortTermMemory.Model(  # swapped: was MultiLayerPerceptron.Model
                input_dim=INPUT_DIM,
                hidden_dim=HIDDEN_DIM,
                step_output_dim=STEP_OUTPUT_DIM,
                future_steps=FUTURE_SAMPLES,
                num_layers=NUM_LAYERS,
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

        self.FutureSeries = TimeSeries(start=0.0, end=0.5, samples=FUTURE_SAMPLES)

        self.Paused = False
        self.Trainer = self.Training()

    def Standalone(self):
        self.Editor = AI4Animation.Scene.AddEntity("Trainer").AddComponent(
            MotionEditor,
            self.Dataset,
            os.path.join(ASSETS_PATH, "Model.glb"),
            BONES,
        )
        AI4Animation.Standalone.Camera.SetTarget(self.Editor.Actor.Entity)
        self.PauseButton = AI4Animation.GUI.Button(
            "Pause Training", 0.4, 0.90, 0.2, 0.04, False, True
        )

    def Update(self):
        if self.Paused:
            return
        try:
            next(self.Trainer)
        except StopIteration:
            pass

    def Training(self):
        for epoch in range(1, EPOCH_COUNT + 1):
            print("Epoch", epoch)
            for xBatch, yBatch in self.DataSampler.SampleBatchesWithinMotions(
                epoch, EPOCH_COUNT
            ):
                _, loss = self.Network.learn(xBatch, yBatch, epoch == 1)
                self.Optimizer.Update(yBatch.shape[0], loss["MSE Loss"])
                for k, v in loss.items():
                    self.LossHistory.Add((Plotting.ToNumpy(v), k))
                yield
            self.LossHistory.Print()

    def GetTrainingFeatures(self, batch):
        motion, timestamps = batch
        mirrored = Tensor.RandomBool()

        inputs = FeedTensor("X", (len(timestamps), INPUT_DIM))
        # FIX 1: sized to FUTURE_SAMPLES * STEP_OUTPUT_DIM instead of flat OUTPUT_DIM
        # so the matrix pivot below can reshape into (batch, FUTURE_SAMPLES, STEP_OUTPUT_DIM)
        outputs = FeedTensor("Y", (len(timestamps), FUTURE_SAMPLES * STEP_OUTPUT_DIM))

        # root = motion.GetModule(RootModule).GetRootTransformations(timestamps, mirrored=mirrored)
        root = Tensor.Inverse(
            motion.GetModule(RootModule).GetTransforms(timestamps, mirrored=mirrored)
        )

        # Inputs
        # transforms = Transform.TransformationTo(
        transforms = Transform.TransformationFrom(
            motion.GetBoneTransformations(timestamps, BONES, mirrored=mirrored),
            root.reshape(-1, 1, 4, 4),
        )
        # velocities = Vector3.DirectionTo(
        velocities = Vector3.DirectionFrom(
            motion.GetBoneVelocities(timestamps, BONES, mirrored=mirrored),
            root.reshape(-1, 1, 4, 4),
        )
        inputs.Feed(Transform.GetPosition(transforms))
        inputs.Feed(Transform.GetAxisZ(transforms))
        inputs.Feed(Transform.GetAxisY(transforms))
        inputs.Feed(velocities)

        # Outputs
        # futureRoot = Transform.TransformationTo(
        futureRoot = Transform.TransformationFrom(
            motion.GetModule(RootModule).GetTransforms(
                self.FutureSeries.SimulateTimestamps(timestamps), mirrored
            ),
            root.reshape(-1, 1, 4, 4),
        )
        # futureMotion = Transform.TransformationTo(
        futureMotion = Transform.TransformationFrom(
            motion.GetModule(MotionModule).GetTransforms(
                self.FutureSeries.SimulateTimestamps(timestamps),
                mirrored,
                BONES,
            ),
            root.reshape(-1, 1, 1, 4, 4),
        )
        outputs.FeedVector3(Transform.GetPosition(futureRoot), x=True, y=False, z=True)
        outputs.FeedVector3(Transform.GetAxisZ(futureRoot), x=True, y=False, z=True)
        outputs.Feed(Transform.GetPosition(futureMotion))
        outputs.Feed(Rotation.GetAxisZ(futureMotion))
        outputs.Feed(Rotation.GetAxisY(futureMotion))

        # =========================================================================
        # FIX 2: MATRIX PIVOT (Framework Feature-Grouped -> LSTM Time-Sequenced)
        # The framework stacks all future samples of each feature contiguously:
        # [root_pos x6, root_fwd x6, motion_pos x6, ...].
        # The LSTM needs features interleaved per timestep:
        # [t0_all_features, t1_all_features, ...].
        # =========================================================================
        Y = outputs.GetTensor()
        batch_size = Y.shape[0]
        B = len(BONES)

        # Slice the grouped feature blocks
        idx = 0
        root_pos = Y[:, idx : idx + FUTURE_SAMPLES * 2].reshape(batch_size, FUTURE_SAMPLES, 2)
        idx += FUTURE_SAMPLES * 2
        root_fwd = Y[:, idx : idx + FUTURE_SAMPLES * 2].reshape(batch_size, FUTURE_SAMPLES, 2)
        idx += FUTURE_SAMPLES * 2
        motion_pos = Y[:, idx : idx + FUTURE_SAMPLES * B * 3].reshape(batch_size, FUTURE_SAMPLES, B * 3)
        idx += FUTURE_SAMPLES * B * 3
        motion_rot_z = Y[:, idx : idx + FUTURE_SAMPLES * B * 3].reshape(batch_size, FUTURE_SAMPLES, B * 3)
        idx += FUTURE_SAMPLES * B * 3
        motion_rot_y = Y[:, idx : idx + FUTURE_SAMPLES * B * 3].reshape(batch_size, FUTURE_SAMPLES, B * 3)

        # Interleave by stacking along the feature dimension for each time step
        Y_interleaved = torch.cat([root_pos, root_fwd, motion_pos, motion_rot_z, motion_rot_y], dim=2)

        # Flatten back to 2D so DataSampler doesn't crash during batch collation
        Y_final = Y_interleaved.reshape(batch_size, FUTURE_SAMPLES * STEP_OUTPUT_DIM)
        # =========================================================================

        return (inputs.GetTensor(), Y_final)

    def GetEditorFeatures(self):
        features = FeedTensor("X", INPUT_DIM)
        root = self.Editor.Actor.Root
        transforms = Transform.TransformationTo(self.Editor.Actor.GetTransforms(), root)
        velocities = Vector3.DirectionTo(self.Editor.Actor.GetVelocities(), root)
        features.Feed(Transform.GetPosition(transforms))
        features.Feed(Transform.GetAxisZ(transforms))
        features.Feed(Transform.GetAxisY(transforms))
        features.Feed(velocities)
        return features.GetTensor()

    def Draw(self):
        self.Network.eval()
        with torch.no_grad():
            xBatch = self.GetEditorFeatures().unsqueeze(0)
            yPred = self.Network(xBatch)

            # =========================================================================
            # FIX 3: MATRIX PIVOT (LSTM Time-Sequenced -> Framework Feature-Grouped)
            # Inverse of FIX 2: unpack per-timestep features back into the contiguous
            # grouped layout that ReadTensor / output.ReadVector3 expects.
            # =========================================================================
            B = len(BONES)
            yPred_seq = yPred.reshape(1, FUTURE_SAMPLES, STEP_OUTPUT_DIM)

            # Extract features back out of their sequential time-steps
            root_pos     = yPred_seq[:, :, 0:2].reshape(1, -1)
            root_fwd     = yPred_seq[:, :, 2:4].reshape(1, -1)
            motion_pos   = yPred_seq[:, :, 4       : 4 + B*3].reshape(1, -1)
            motion_rot_z = yPred_seq[:, :, 4 + B*3 : 4 + B*6].reshape(1, -1)
            motion_rot_y = yPred_seq[:, :, 4 + B*6 : 4 + B*9].reshape(1, -1)

            # Concatenate into the flat grouped layout ReadTensor expects
            yPred_grouped = torch.cat([root_pos, root_fwd, motion_pos, motion_rot_z, motion_rot_y], dim=1)

            # squeeze(0) removes the batch dim so ReadTensor receives a flat 1D array
            output = ReadTensor("Y", Tensor.ToNumPy(yPred_grouped.squeeze(0)))
            # =========================================================================

            root = self.Editor.Actor.Root

            # Trajectory
            futureRoot = Transform.TransformationFrom(
                Transform.TR(
                    output.ReadVector3(FUTURE_SAMPLES, True, False, True),
                    Rotation.Look(
                        output.ReadVector3(FUTURE_SAMPLES, True, False, True),
                        Vector3.UnitY(FUTURE_SAMPLES),  # must match FUTURE_SAMPLES rows for Rotation.Look shape
                    ),
                ),
                root,
            )
            rootSeries = RootModule.Series(self.FutureSeries, futureRoot)
            rootSeries.Draw()

            # Motion
            futureMotion = Transform.TransformationFrom(
                Transform.TR(
                    output.ReadVector3((FUTURE_SAMPLES, len(BONES))),
                    output.ReadRotation3D((FUTURE_SAMPLES, len(BONES))),
                ),
                root,
            )
            motionSeries = MotionModule.Series(self.FutureSeries, BONES, futureMotion)
            motionSeries.Draw()

        self.Network.train()

    def GUI(self):
        self.PauseButton.GUI()
        self.Paused = self.PauseButton.Active


def main():
    AI4Animation(Program(), mode=AI4Animation.Mode.HEADLESS)


if __name__ == "__main__":
    main()