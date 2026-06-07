# ============================================================================
# GUI INFERENCE - Flow Matching ston LATENT xwro (Erwthma 5, meros B)
#
# Fortwnei (1) ton VAE 'vae_full_model.pth' kai (2) to apothikeumeno
# 'flow_matching_latent_model.pth' kai trexei to grafiko perivallon (STANDALONE)
# XWRIS ekpaideush. Prepei prwta na exei trexei to ProgramFlowMatchingLatent.py.
#
# Roh sto Draw: trexon kare -> encode se latent -> Flow Matching provlepei to
# epomeno latent -> VAE decode -> raw kinhsh ston kokkino Actor.
# ============================================================================
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

from ai4animation.AI.Models import FlowMatching  # noqa: F401
from ai4animation.AI.Models.AutoencoderVAE import VAEAutoencoder  # noqa: F401

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/Cranberry")

sys.path.append(ASSETS_PATH)
import Definitions

FRAMERATE = 30
BONES = Definitions.FULL_BODY_NAMES
FRAME_DIM = 12 * len(BONES)
LATENT_DIM = 256
WINDOW_SIZE = 5
FLOW_STEPS = 10
VAE_PATH = "vae_full_model.pth"
MODEL_PATH = "flow_matching_latent_model.pth"


def _find_model(path):
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

        # ---- VAE ----
        vae_file = _find_model(VAE_PATH)
        print(f"Loading VAE from '{vae_file}'...")
        self.VAE = torch.load(vae_file, weights_only=False)
        self.VAE = Tensor.ToDevice(self.VAE)
        self.VAE.eval()
        for p in self.VAE.parameters():
            p.requires_grad = False

        # ---- Flow Matching (latent) ----
        model_file = _find_model(MODEL_PATH)
        print(f"Loading trained latent Flow Matching model from '{model_file}'...")
        self.Network = torch.load(model_file, weights_only=False)
        self.Network = Tensor.ToDevice(self.Network)
        self.Network.eval()
        print("Models loaded! Running latent-generation inference in the GUI.")

        self.EditorHistory = torch.zeros(1, WINDOW_SIZE, LATENT_DIM)

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
            cond = self.EditorHistory.reshape(1, WINDOW_SIZE * LATENT_DIM)

            predicted_latent = self.Network(cond, steps=FLOW_STEPS)

            predicted_raw_norm = self.VAE.Decoder(predicted_latent)
            predicted_raw = self.VAE.Statistics.Denormalize(predicted_raw_norm)

            yPred = Tensor.ToNumPy(predicted_raw)
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
