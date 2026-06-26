import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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
from ai4animation.AI.Models import AutoregressionMLP

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/CranberryInf")
MODEL_PATH = str(SCRIPT_DIR.parent.parent.parent)
sys.path.append(ASSETS_PATH)
import Definitions

EPOCH_COUNT = 150 
BATCH_SIZE = 32 
FRAMERATE = 30 
DRAW_INTERVAL = 500 
BONES = Definitions.FULL_BODY_NAMES 
FRAME_DIM = 12 * len(BONES) 
HIDDEN_DIM = 512 
WINDOW_SIZE = 5 


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

        save_path = os.path.join(MODEL_PATH, "AutoregressiveMLP_Model.pth")
        try:
            os.path.exists(save_path)
            print(f"Found trained model. Loading full model from: {save_path}")
            self.Network = torch.load(save_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=False)
        except:
            print("Model {save_path} not found")

        
        self.HistoryOffsets = torch.arange(-WINDOW_SIZE, 0) / FRAMERATE

        
        self.EditorHistory = torch.zeros(1, WINDOW_SIZE, FRAME_DIM)

    
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


    def GetEditorFeatures(self):
        features = FeedTensor("X", FRAME_DIM)
        root = self.Editor.Actor.Root
        
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
            current_frame = self.GetEditorFeatures().unsqueeze(0).unsqueeze(1) 

            
            self.EditorHistory = torch.cat([self.EditorHistory[:, 1:, :], current_frame], dim=1) 
            
            
            yPred = Tensor.ToNumPy(self.Network(self.EditorHistory, generate_steps=1))
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
    AI4Animation(Program(), mode=AI4Animation.Mode.STANDALONE)


if __name__ == "__main__":
    main()
