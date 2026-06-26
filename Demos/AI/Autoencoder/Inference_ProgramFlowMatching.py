import os
import sys
from pathlib import Path

import torch
from ai4animation import (
    Actor,
    AI4Animation,
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

# kaneis import to montelo wste na mporei to torch.load na to anagnwrisei
from ai4animation.AI.Models import FlowMatching 

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/CranberryInf")
MODEL_PATH = str(SCRIPT_DIR.parent.parent.parent)
sys.path.append(ASSETS_PATH)
import Definitions

FRAMERATE = 30
BONES = Definitions.FULL_BODY_NAMES
FRAME_DIM = 12 * len(BONES)
WINDOW_SIZE = 5
FLOW_STEPS = 10
MODEL_PATH = os.path.join(MODEL_PATH, "flow_matching_raw_model.pth")


def _find_model(path):
    # psaxnei to montelo sto cwd kai se merika logika monopatia
    candidates = [path, os.path.join(os.getcwd(), path), str(SCRIPT_DIR / path)]
    for c in candidates:
        if os.path.exists(c):
            return c
    return path


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

        model_file = _find_model(MODEL_PATH)
        print(f"Loading trained Flow Matching model from '{model_file}'...")
        self.Network = torch.load(model_file, weights_only=False)
        self.Network = Tensor.ToDevice(self.Network)
        self.Network.eval()
        print("Model loaded! Running inference in the GUI.")

        self.EditorHistory = torch.zeros(1, WINDOW_SIZE, FRAME_DIM)

    def Standalone(self):
        entity = AI4Animation.Scene.AddEntity("Editor")
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
        # KAMIA ekpaideush - mono inference
        pass

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
            current_frame = self.GetEditorFeatures().unsqueeze(0).unsqueeze(1)
            self.EditorHistory = torch.cat([self.EditorHistory[:, 1:, :], current_frame], dim=1)

            cond = self.EditorHistory.reshape(1, WINDOW_SIZE * FRAME_DIM)
            yPred = Tensor.ToNumPy(self.Network(cond, steps=FLOW_STEPS))
            output = ReadTensor("Y", yPred)

            self.Actor.Root = self.Editor.Actor.Root
            self.Actor.SetPositions(Vector3.PositionFrom(output.ReadVector3(len(BONES)), self.Actor.Root))
            self.Actor.SetRotations(Rotation.RotationFrom(output.ReadRotation3D(len(BONES)), self.Actor.Root))
            self.Actor.SetVelocities(Vector3.DirectionFrom(output.ReadVector3(len(BONES)), self.Actor.Root))
            for bone in self.Actor.Bones:
                bone.RestoreLength()
            self.Actor.RestoreBoneAlignments()
            self.Actor.SyncToScene()


def main():
    AI4Animation(Program(), mode=AI4Animation.Mode.STANDALONE)


if __name__ == "__main__":
    main()
