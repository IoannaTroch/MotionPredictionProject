# program for autoregressive MLP
# predicts the next frame of motion based on past frames
# uses a sliding window containing a number of past frames
import os
import sys
from pathlib import Path
import numpy as np

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
from ai4animation.AI.Models import AutoregressionMLP

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/Cranberry")

sys.path.append(ASSETS_PATH)
import Definitions

EPOCH_COUNT = 150 
BATCH_SIZE = 32 # animation snapshots processed simultaneously
FRAMERATE = 30 # frames per second
DRAW_INTERVAL = 500 # how often to display plots
BONES = Definitions.FULL_BODY_NAMES # array of strings representing joints
FRAME_DIM = 12 * len(BONES) # every joint has 12 dimensions
# position 3, forward 3, upward 3, velocity 3
HIDDEN_DIM = 512 # neurons in the hidden layers
WINDOW_SIZE = 5 # it looks at the past 5 frames


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
                lambda x: MotionModule(x), # tracks joint rotations 
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
            AutoregressionMLP.Model( # sets the model with default dropout
                FRAME_DIM, WINDOW_SIZE, HIDDEN_DIM
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

        # generates a relative history index matrix 
        self.HistoryOffsets = torch.arange(-WINDOW_SIZE, 0) / FRAMERATE

        # sliding history window
        self.EditorHistory = torch.zeros(1, WINDOW_SIZE, FRAME_DIM)

        self.Trainer = self.Training()

    # generates the scene and actor for the 3d environment
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
            for batch_data in self.DataSampler.SampleBatchesWithinMotions(
                epoch, EPOCH_COUNT
            ):
                xBatch = batch_data[0]
                yBatch = batch_data[1]
                loss = self.Network.learn(xBatch, yBatch, epoch == 1) # forward and backward pass
                self.Optimizer.Update(yBatch.shape[0], loss["MSE Loss"])
                for k, v in loss.items():
                    self.LossHistory.Add((Plotting.ToNumpy(v), k))
                yield # pauses for syncing
            self.LossHistory.Print()

    # generates tensor with frame features
    # same as part of GetTrainingFeatures() in Demos/Autoencoder/Program.py
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

        if isinstance(timestamps, np.ndarray): # fixes error with unsqueeze
            timestamps = torch.from_numpy(timestamps)

        mirrored = Tensor.RandomBool() # random mirroring for data augmentation

        # calculates the inverse root transformation matrix (normalizes tracking space context)
        root = Tensor.Inverse(
            motion.GetModule(RootModule).GetTransforms(timestamps, mirrored=mirrored)
        )

        frames = self.ExtractFrameFeatures(motion, timestamps, mirrored, root) # target frames

        # does the same based on history 
        history_timestamps = timestamps.unsqueeze(-1) + self.HistoryOffsets.to(timestamps.device)
        history_flat = history_timestamps.flatten()

        history_roots = Tensor.Inverse(motion.GetModule(RootModule).GetTransforms(history_flat, mirrored=mirrored))
        history_frames_flat = self.ExtractFrameFeatures(motion, history_flat, mirrored, history_roots) # history frames

        x_history_windows = history_frames_flat.reshape(len(timestamps), WINDOW_SIZE, FRAME_DIM) # reshapes history data
        
        x_history_flattened = x_history_windows.reshape(len(timestamps), WINDOW_SIZE * FRAME_DIM) # flattens data for the next layer

        return (x_history_flattened, frames)

    def GetEditorFeatures(self):
        features = FeedTensor("X", FRAME_DIM)
        root = self.Editor.Actor.Root
        # normalise 3d editor coords to match root orientation layout logic
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
        with torch.no_grad():
            current_frame = self.GetEditorFeatures().unsqueeze(0).unsqueeze(1) # matches dimensions with yPred # [1, 1, FRAME_DIM]

            # deletes the first frame from the history window and adds current frame 
            self.EditorHistory = torch.cat([self.EditorHistory[:, 1:, :], current_frame], dim=1) 
            
            # generates one prediction frame based on the updated history window
            yPred = Tensor.ToNumPy(self.Network(self.EditorHistory, generate_steps=1))
            output = ReadTensor("Y", yPred)

            self.Actor.Root = self.Editor.Actor.Root # snaps the prediction actor's core location to match the master trajectory root
            self.Actor.SetPositions( # changes character's bone arrays based on parsed network output features
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
