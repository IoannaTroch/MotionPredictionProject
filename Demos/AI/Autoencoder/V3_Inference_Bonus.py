# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
import math
import numpy as np
from pathlib import Path
import torch

from ai4animation import (
    Actor, AI4Animation, LongShortTermMemory, Dataset, FeedTensor,
    MirrorModule, MotionEditor, MotionModule, ReadTensor, RootModule,
    Rotation, Tensor, Transform, Utility, Vector3
)
from ai4animation.AI.Models.AutoencoderVAE import VAEAutoencoder

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/Cranberry")
sys.path.append(ASSETS_PATH)
import Definitions

BONES = Definitions.FULL_BODY_NAMES
FRAME_DIM = 12 * len(BONES)
LATENT_DIM = 256
HIDDEN_DIM = 512

# ΠΡΕΠΕΙ ΝΑ ΤΑΙΡΙΑΖΟΥΝ ΑΠΟΛΥΤΑ ΜΕ ΤΗΝ ΕΚΠΑΙΔΕΥΣΗ
WINDOW_SIZE = 15 
NUM_STYLES = 5 
CONDITION_MULTIPLIER = 10 

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

        print("Loading pre-trained VAE...")
        self.VAE = torch.load("vae_full_model.pth", weights_only=False)
        self.VAE = Tensor.ToDevice(self.VAE)
        self.VAE.eval() 

        print("Loading Conditional LSTM...")
        self.Network = torch.load("conditional_lstm_full.pth", weights_only=False)
        self.Network = Tensor.ToDevice(self.Network)
        self.Network.eval()

        self.EditorHistory = torch.zeros(1, WINDOW_SIZE, LATENT_DIM)
        self.CurrentStyle = 0
        
        # ========================================================
        # ΤΟ ΔΙΚΟ ΜΑΣ "GPS" (Ανεξάρτητη Πλοήγηση Τρίτου Προσώπου)
        # ========================================================
        self.MyPosition = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.MyRotationY = 0.0 # Σε ακτίνια (radians)

        print("\n" + "="*50)
        print("ΤΟ ΣΥΣΤΗΜΑ ΕΙΝΑΙ ΕΤΟΙΜΟ!")
        print("- ΚΙΝΗΣΗ (Navigation): Βελάκια (Πάνω, Κάτω, Αριστερά, Δεξιά)")
        print("- ΣΤΥΛ (Animation): Πλήκτρα 1 (Stand), 2 (WalkFwd), 3 (WalkBwd), 4 (Crouch), 5 (VR)")
        print("="*50)

    def Standalone(self):
        # Το Dataset Dummy (Γκρι) - Κρύψτο από το GUI
        entity = AI4Animation.Scene.AddEntity("Trainer")
        self.Editor = entity.AddComponent(MotionEditor, self.Dataset, os.path.join(ASSETS_PATH, "Model.glb"), BONES)
        
        # Το δικό μας Μοντέλο AI (Κόκκινο)
        self.Actor = AI4Animation.Scene.AddEntity("Actor").AddComponent(Actor, os.path.join(ASSETS_PATH, "Model.glb"), BONES)
        self.Actor.SkinnedMesh.SetColor(AI4Animation.Color.RED)
        AI4Animation.Standalone.Camera.SetTarget(self.Actor.Entity)

    def Update(self):
        pass # Χωρίς εκπαίδευση εδώ

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
        import pyray as rl
        
        # -----------------------------------------------------
        # 1. ΕΛΕΓΧΟΣ ΣΤΥΛ (ANIMATION) ΜΕ ΑΡΙΘΜΟΥΣ
        # -----------------------------------------------------
        if rl.is_key_pressed(rl.KEY_ONE): self.CurrentStyle = 0
        elif rl.is_key_pressed(rl.KEY_TWO): self.CurrentStyle = 1
        elif rl.is_key_pressed(rl.KEY_THREE): self.CurrentStyle = 2
        elif rl.is_key_pressed(rl.KEY_FOUR): self.CurrentStyle = 3
        elif rl.is_key_pressed(rl.KEY_FIVE): self.CurrentStyle = 4

        # -----------------------------------------------------
        # 2. ΕΛΕΓΧΟΣ ΚΙΝΗΣΗΣ (NAVIGATION) ΜΕ ΒΕΛΑΚΙΑ
        # -----------------------------------------------------
        turn_speed = 0.05
        move_speed = 0.06
        
        if rl.is_key_down(rl.KEY_LEFT):
            self.MyRotationY += turn_speed
        if rl.is_key_down(rl.KEY_RIGHT):
            self.MyRotationY -= turn_speed
            
        forward_dir = np.array([math.sin(self.MyRotationY), 0.0, math.cos(self.MyRotationY)])
        
        if rl.is_key_down(rl.KEY_UP):
            self.MyPosition += forward_dir * move_speed
        if rl.is_key_down(rl.KEY_DOWN):
            self.MyPosition -= forward_dir * move_speed

        c = math.cos(self.MyRotationY)
        s = math.sin(self.MyRotationY)
        
        self.MyRoot = np.identity(4, dtype=np.float32)
        self.MyRoot[0, 0] = c
        self.MyRoot[0, 2] = s
        self.MyRoot[2, 0] = -s
        self.MyRoot[2, 2] = c
        self.MyRoot[0, 3] = self.MyPosition[0]
        self.MyRoot[1, 3] = self.MyPosition[1]
        self.MyRoot[2, 3] = self.MyPosition[2]

        # -----------------------------------------------------
        # 3. ΥΠΟΛΟΓΙΣΜΟΙ ΝΕΥΡΩΝΙΚΟΥ ΔΙΚΤΥΟΥ (INFERENCE)
        # -----------------------------------------------------
        with torch.no_grad():
            current_frame_raw = self.GetEditorFeatures().unsqueeze(0)
            current_latent = self.EncodeBatch(current_frame_raw, 1).unsqueeze(1)
            history_flat = self.EditorHistory.reshape(1, WINDOW_SIZE * LATENT_DIM)

            user_condition = torch.zeros(1, NUM_STYLES)
            user_condition[0, self.CurrentStyle] = 1.0
            
            user_condition_amplified = user_condition.repeat(1, CONDITION_MULTIPLIER)
            user_condition_amplified = Tensor.ToDevice(user_condition_amplified)

            history_conditional = torch.cat([history_flat, user_condition_amplified], dim=1)
            predicted_latent = self.Network(history_conditional)

            # Δίχτυ Ασφαλείας
            predicted_latent = torch.clamp(predicted_latent, min=-4.0, max=4.0)

            # Ανανέωση Ιστορικού (100% Closed Loop Αυτονομία)
            self.EditorHistory = torch.cat([self.EditorHistory[:, 1:, :], predicted_latent.unsqueeze(1)], dim=1)

            # Αποκωδικοποίηση VAE
            predicted_raw_norm = self.VAE.Decoder(predicted_latent)
            predicted_raw = self.VAE.Statistics.Denormalize(predicted_raw_norm)

            yPred = Tensor.ToNumPy(predicted_raw)
            output = ReadTensor("Y", yPred)

            # 4. Εφαρμογή της κίνησης ΠΑΝΩ ΣΤΟ ΔΙΚΟ ΜΑΣ GPS
            self.Actor.Root = self.MyRoot
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