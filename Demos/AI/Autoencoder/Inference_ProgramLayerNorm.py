# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
from pathlib import Path
import numpy as np
import torch

from ai4animation import (
    Actor, AI4Animation, Dataset, FeedTensor, MirrorModule, 
    MotionEditor, MotionModule, ReadTensor, RootModule, 
    Rotation, Tensor, Transform, Utility, Vector3
)

##fortwnw to patch me to validation dataset
from ai4animation.AI.Models.AutoencoderLayerNorm import EnchancedAutoencoder

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/Cranberry")
sys.path.insert(0, ASSETS_PATH) 
import Definitions

BONES = Definitions.FULL_BODY_NAMES
FRAME_DIM = 12 * len(BONES)

class Program:
    def Start(self):
        Utility.SetSeed(23456)
        
        self.Dataset = Dataset(
            os.path.join(ASSETS_PATH, "Motions"),
            [
                lambda x: RootModule(x, Definitions.HipName, Definitions.LeftHipName, Definitions.RightHipName, Definitions.LeftShoulderName, Definitions.RightShoulderName, Definitions.NeckName),
                lambda x: MotionModule(x),
                lambda x: MirrorModule(x, Vector3.Axis.ZPositive, Vector3.Create(0, 0, 180)),
            ],
        )

        print("Loading pre-trained LayerNorm Autoencoder...")
        #fortwnw to trained montelo
        self.Network = torch.load("layernorm_full_model.pth", weights_only=False)
        self.Network = Tensor.ToDevice(self.Network)
        self.Network.eval() # Σημαντικό: Κλείνει τα τυχόν dropouts/train modes

        print("\n" + "="*50)
        print("ΤΟ LAYER NORM ΜΟΝΤΕΛΟ ΦΟΡΤΩΘΗΚΕ ΕΠΙΤΥΧΩΣ!")
        print("Αν η εκπαίδευση πήγε καλά, ο ΚΟΚΚΙΝΟΣ χαρακτήρας")
        print("πρέπει να ακολουθεί πιστά τον ΚΑΝΟΝΙΚΟ χαρακτήρα.")
        print("="*50)

    def Standalone(self):
        entity = AI4Animation.Scene.AddEntity("Trainer")
        
        #xarakthras animation apo dataset
        self.Editor = entity.AddComponent(MotionEditor, self.Dataset, os.path.join(ASSETS_PATH, "Model.glb"), BONES)
        
        #xarakthraw animation apo montelo
        self.Actor = AI4Animation.Scene.AddEntity("Actor").AddComponent(Actor, os.path.join(ASSETS_PATH, "Model.glb"), BONES)
        self.Actor.SkinnedMesh.SetColor(AI4Animation.Color.RED)
        
        AI4Animation.Standalone.Camera.SetTarget(self.Actor.Entity)

    def Update(self): 
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
            current_frame_raw = self.GetEditorFeatures().unsqueeze(0) 
            predicted_raw = self.Network(current_frame_raw)
            yPred = Tensor.ToNumPy(predicted_raw.squeeze(0))
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