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
from ai4animation.AI.Models.AutoencoderVAE import VAEAutoencoder

SCRIPT_DIR = Path(__file__).parent
#fortwnw to patch me to va;idation dataset
ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/CranberryInf")
sys.path.insert(0, ASSETS_PATH) 
import Definitions

BONES = Definitions.FULL_BODY_NAMES
FRAME_DIM = 12 * len(BONES)

FRAMERATE = 30 
BONES = Definitions.FULL_BODY_NAMES 
FRAME_DIM = 12 * len(BONES) 
LATENT_DIM = 256        
WINDOW_SIZE = 5 

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

        print("Φόρτωση pre-trained VAE...")
        if os.path.exists("vae_full_model.pth"):
            self.VAE = torch.load("vae_full_model.pth", weights_only=False)
            self.VAE = Tensor.ToDevice(self.VAE)
            self.VAE.eval()
            print("Το μοντέλο VAE φορτώθηκε επιτυχώς!")
        else:
            print("ΣΦΑΛΜΑ: Το 'vae_full_model.pth' δε βρέθηκε! Δε γίνεται να προχωρήσει η εκτέλεση χωρίς το VAE.")
            sys.exit()

        #fortwsh montelou
        print("Φόρτωση pre-trained Latent LSTM...")
        if os.path.exists("latent_lstm_layernorm_full_model.pth"):
            self.Network = torch.load("latent_lstm_layernorm_full_model.pth", weights_only=False)
            self.Network = Tensor.ToDevice(self.Network)
            self.Network.eval() 
            print("Το μοντέλο Latent LSTM φορτώθηκε επιτυχώς!")
        else:
            print("ΣΦΑΛΜΑ: Το 'latent_lstm_with_vae_full_model.pth' δε βρέθηκε!")
            sys.exit()

        self.EditorHistory = Tensor.ToDevice(torch.zeros(1, WINDOW_SIZE, LATENT_DIM))

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
        pass

    def EncodeBatch(self, raw_data, window_size):
        batch_size = raw_data.shape[0]
        raw_reshaped = raw_data.view(-1, FRAME_DIM)

        raw_tensor = Tensor.ToDevice(raw_reshaped.clone().detach().float())
        
        norm_raw = self.VAE.Statistics.Normalize(raw_tensor)
        hidden = self.VAE.EncoderBody(norm_raw)
        mu = self.VAE.fc_mu(hidden)
        
        return mu.view(batch_size, window_size * LATENT_DIM)

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

            self.EditorHistory = torch.cat([self.EditorHistory[:, 1:, :], current_latent], dim=1) 
            history_flat = self.EditorHistory.reshape(1, WINDOW_SIZE * LATENT_DIM)
 
            predicted_latent = self.Network(history_flat) 
            
            predicted_raw_norm = self.VAE.Decoder(predicted_latent)
            predicted_raw = self.VAE.Statistics.Denormalize(predicted_raw_norm)
            
            yPred = Tensor.ToNumPy(predicted_raw)
            output = ReadTensor("Y", yPred)
        
            # Εφάρμοσε την πρόβλεψη πάνω στον χαρακτήρα (Actor)
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