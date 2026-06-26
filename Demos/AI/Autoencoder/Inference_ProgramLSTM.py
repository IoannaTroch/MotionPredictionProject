# uses LSTM model to train an autoregressive LSTM with a window of a few frames
# to predict the next frame of motion
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
from ai4animation import (
    Actor,
    AI4Animation,
    LongShortTermMemory,
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
ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/CranberryInf")
MODEL_PATH = str(SCRIPT_DIR.parent.parent.parent)

sys.path.append(ASSETS_PATH)
import Definitions

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

        # loading model if saved
        save_path = os.path.join(MODEL_PATH, "Autoregressive_LSTM_Model.pth")
        try:
            os.path.exists(save_path)
            print(f"Found trained model. Loading full model from: {save_path}")
            self.Network = torch.load(save_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=False)
        except:
            print("Model {save_path} not found")

        self.LossHistory = Plotting.LossHistory(
            "Loss History", drawInterval=DRAW_INTERVAL, yScale="log"
        )

        # generates a relative history index matrix 
        self.HistoryOffsets = torch.arange(-WINDOW_SIZE, 0) / FRAMERATE

        # sliding history window
        self.EditorHistory = torch.zeros(1, WINDOW_SIZE, FRAME_DIM)

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
        pass

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

    # def GetTrainingFeatures(self, batch):
    #     motion, timestamps = batch

    #     if isinstance(timestamps, np.ndarray): # fixes error with unsqueeze
    #         timestamps = torch.from_numpy(timestamps)

    #     mirrored = Tensor.RandomBool() # random mirroring for data augmentation

    #     # calculates the inverse root transformation matrix (normalizes tracking space context)
    #     root = Tensor.Inverse(
    #         motion.GetModule(RootModule).GetTransforms(timestamps, mirrored=mirrored)
    #     )

    #     frames = self.ExtractFrameFeatures(motion, timestamps, mirrored, root) # target frames

    #     # does the same based on history 
    #     history_timestamps = timestamps.unsqueeze(-1) + self.HistoryOffsets.to(timestamps.device)
    #     history_flat = history_timestamps.flatten()

    #     history_roots = Tensor.Inverse(motion.GetModule(RootModule).GetTransforms(history_flat, mirrored=mirrored))
    #     history_frames_flat = self.ExtractFrameFeatures(motion, history_flat, mirrored, history_roots) # history frames

    #     x_history_windows = history_frames_flat.reshape(len(timestamps), WINDOW_SIZE, FRAME_DIM) # reshapes history data
        
    #     x_history_flattened = x_history_windows.reshape(len(timestamps), WINDOW_SIZE * FRAME_DIM) # flattens data for the next layer

    #     return (x_history_flattened, frames)

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
            
            # Flattens the [1, 10, 324] window into a [1, 3240] array for the LSTM
            history_flat = self.EditorHistory.reshape(1, WINDOW_SIZE * FRAME_DIM)
            
            # Generates one prediction frame
            yPred = Tensor.ToNumPy(self.Network(history_flat))
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
    AI4Animation(Program(), mode=AI4Animation.Mode.STANDALONE)


if __name__ == "__main__":
    main()
