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

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/100style")
MODEL_PATH = str(SCRIPT_DIR.parent.parent.parent)
MODEL_NAME = "100style_character.glb" # "Model.glb"

sys.path.append(ASSETS_PATH)
import Definitions

EPOCH_COUNT = 0 # set to 0 if you want to load the model and skip training
BATCH_SIZE = 32 # animation snapshots processed simultaneously
FRAMERATE = 30 # frames per second
DRAW_INTERVAL = 500 # how often to display plots
BONES = Definitions.FULL_BODY_NAMES # array of strings representing joints

# CONDITIONAL DIMENSIONS (Bonus)

FRAME_DIM = 12 * len(BONES) # every joint has 12 dimensions
# position 3, forward 3, upward 3, velocity 3
CONDITION_DIM = 2 # [1, 0] for Neutral (Walking), [0, 1] for Target (Crouching)
INPUT_DIM = FRAME_DIM + CONDITION_DIM # Total input to the network


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
            LongShortTermMemory.Model(
                input_dim=WINDOW_SIZE * INPUT_DIM, 
                output_dim=FRAME_DIM,              
                hidden_dim=HIDDEN_DIM,
                future_steps=1,                    
                num_layers=2
            )
        )
        
        # loading model if saved
        save_path = os.path.join(MODEL_PATH, "Autoregressive_LSTM_Model_Bonus.pth")
        if os.path.exists(save_path) and EPOCH_COUNT == 0:
            print(f"--> Found trained model! Loading FULL MODEL from: {save_path}")
            self.Network = torch.load(save_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=False)

        self.LossHistory = Plotting.LossHistory(
            "Loss History", drawInterval=DRAW_INTERVAL, yScale="log"
        )

        # generates a relative history index matrix 
        self.HistoryOffsets = torch.arange(-WINDOW_SIZE, 0) / FRAMERATE

        # sliding history window (now includes condition padding)
        self.EditorHistory = torch.zeros(1, WINDOW_SIZE, INPUT_DIM)

        self.Trainer = self.Training()

    # generates the scene and actor for the 3d environment
    def Standalone(self): 
        entity = AI4Animation.Scene.AddEntity("Trainer")
        self.Editor = entity.AddComponent(
            MotionEditor,
            self.Dataset,
            os.path.join(ASSETS_PATH, "MODEL_NAME"),
            BONES,
        )
        self.Actor = AI4Animation.Scene.AddEntity("Actor").AddComponent(
            Actor, os.path.join(ASSETS_PATH, "MODEL_NAME"), BONES
        )
        self.Actor.SkinnedMesh.SetColor(AI4Animation.Color.RED)
        AI4Animation.Standalone.Camera.SetTarget(self.Actor.Entity)

        # REAL-TIME UI BUTTON (Moved to Top-Right)
        self.StyleButton = AI4Animation.GUI.Button(
            "Style: Walking (Neutral)", 0.80, 0.05, 0.15, 0.04, False, True
        )
    #     # generates the scene and actor for the 3d environment
    # def Standalone(self): 
    #     entity = AI4Animation.Scene.AddEntity("Trainer")
    #     self.Editor = entity.AddComponent(
    #         MotionEditor,
    #         self.Dataset,
    #         os.path.join(ASSETS_PATH, "Model.glb"),
    #         BONES,
    #     )
        
    #     # =========================================================================
    #     # THE GHOST FIX: Make the White Dummy completely invisible
    #     # It will still drive the math, but won't clutter the screen!
    #     # =========================================================================
    #     self.Editor.Actor.SkinnedMesh.SetVisibility(False) 

    #     self.Actor = AI4Animation.Scene.AddEntity("Actor").AddComponent(
    #         Actor, os.path.join(ASSETS_PATH, "Model.glb"), BONES
    #     )
    #     self.Actor.SkinnedMesh.SetColor(AI4Animation.Color.RED)
    #     AI4Animation.Standalone.Camera.SetTarget(self.Actor.Entity)

    #     # REAL-TIME UI BUTTON
    #     self.StyleButton = AI4Animation.GUI.Button(
    #         "Style: Walking (Neutral)", 0.05, 0.05, 0.15, 0.04, False, True
    #     )

    def GUI(self):
        self.StyleButton.GUI()
        if self.StyleButton.Active:
            self.StyleButton.Label = "Style: Crouching (Target)"
        else:
            self.StyleButton.Label = "Style: Walking (Neutral)"

    def Update(self):
        try:
            next(self.Trainer)
        except StopIteration as e:
            pass

    def Training(self): 
        if EPOCH_COUNT==0: # used when loading a trained model
            print("\n*** INFERENCE MODE: Skipping training and opening Viewer! ***\n")
            return
        
        print("Splitting dataset into Training (80%) and Validation (20%)...")
        all_batches = list(self.DataSampler.SampleBatchesWithinMotions(1, EPOCH_COUNT))
        
        total_batches = len(all_batches)
        split_idx = int(0.8 * total_batches) 
        
        train_batches = all_batches[:split_idx]
        val_batches = all_batches[split_idx:]
        
        # upologizw sunoliko athroisma twn deigmatwn se ola ta training batches
        total_train_samples = sum([batch_data[1].shape[0] for batch_data in train_batches])
        
        self.Optimizer = Utility.CosineAnnealingOptimizer(
            self.Network.parameters(),
            self.DataSampler.BatchSize,
            total_train_samples
        )
        
        print(f"Total batches: {total_batches} | Train: {len(train_batches)} | Val: {len(val_batches)}")

        train_losses_history = []
        val_losses_history = []
        best_val_loss = float('inf')
        
        for epoch in range(1, EPOCH_COUNT + 1):
            print(f"\n--- Epoch {epoch}/{EPOCH_COUNT} ---")
            
            # training
            self.Network.train()
            epoch_train_loss = 0.0
            
            for i, batch_data in enumerate(train_batches):
                xBatch = batch_data[0]
                yBatch = batch_data[1]
                
                _, loss = self.Network.learn(xBatch, yBatch, epoch == 1) 
                
                if isinstance(loss, dict):
                    tensor_loss = sum(loss.values())
                else:
                    tensor_loss = loss
                
                self.Optimizer.Update(yBatch.shape[0], tensor_loss) 
                epoch_train_loss += tensor_loss.item()
                
                progress = 100 * (i + 1) / len(train_batches)
                print(f"Training Progress: {progress:.1f}%", end="\r")
                yield
            print(" " * 50, end="\r")
                
            avg_train_loss = epoch_train_loss / len(train_batches)
            train_losses_history.append(avg_train_loss)
            
            # VALIDATIONNN
            self.Network.eval() 
            epoch_val_loss = 0.0
            with torch.no_grad(): 
                for i, batch_data in enumerate(val_batches):
                    xBatch = batch_data[0]
                    yBatch = batch_data[1]
                    
                    _, loss = self.Network.learn(xBatch, yBatch, update_statistics=False)

                    if isinstance(loss, dict):
                        tensor_loss = sum(v.item() if hasattr(v, 'item') else v for v in loss.values())
                    else:
                        tensor_loss = loss.item() if hasattr(loss, 'item') else loss
                    
                    epoch_val_loss += tensor_loss
                    
                    progress = 100 * (i + 1) / len(val_batches)
                    print(f"Validation Progress: {progress:.1f}%", end="\r")
                    yield
            print(" " * 50, end="\r")
            
            avg_val_loss = epoch_val_loss / len(val_batches)
            val_losses_history.append(avg_val_loss)
            
            print(f"Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(">>> New best validation loss! <<<")
                
            self.PlotTrainVal(train_losses_history, val_losses_history, epoch)
            
        plt.ioff() 
        plt.savefig("loss_history_LSTM.png", dpi=300, bbox_inches='tight') 
        
        # saves the model 
        save_path = os.path.join(MODEL_PATH, "Autoregressive_LSTM_Model_Bonus.pth")
        torch.save(self.Network, save_path)
        print(f"The model was saved successfully to {save_path}!")
        plt.show() 

    # functions for plotting training and validation loss
    def PlotTrainVal(self, train_losses, val_losses, epoch):
        plt.ion() 
        plt.clf()
        
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue', linewidth=2)
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='orange', linewidth=2)
        
        plt.title(f'Conditional Autoregressive LSTM MSE Loss (Epoch {epoch}/{EPOCH_COUNT})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Log Scale)')
        plt.yscale('log') 
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        
        plt.pause(0.01)

    # generates tensor with frame features
    # same as part of GetTrainingFeatures() in Demos/Autoencoder/Program.py
    def ExtractFrameFeatures(self, motion, timestamps, mirrored, root, is_target_style=False):
        transforms = Transform.TransformationFrom(
            motion.GetBoneTransformations(timestamps, BONES, mirrored=mirrored),
            root.reshape(-1, 1, 4, 4),
        )
        velocities = Vector3.DirectionFrom(
            motion.GetBoneVelocities(timestamps, BONES, mirrored=mirrored),
            root.reshape(-1, 1, 4, 4), 
        )
        
        inputs = FeedTensor("Frame", (len(timestamps), INPUT_DIM))
        inputs.Feed(Transform.GetPosition(transforms)) 
        inputs.Feed(Transform.GetAxisZ(transforms)) 
        inputs.Feed(Transform.GetAxisY(transforms)) 
        inputs.Feed(velocities) 

        cond_array = np.zeros((len(timestamps), CONDITION_DIM), dtype=np.float32)
        if is_target_style:
            cond_array[:, 1] = 1.0 
        else:
            cond_array[:, 0] = 1.0 
        inputs.Feed(cond_array)

        return inputs.GetTensor()

    def GetTrainingFeatures(self, batch):
        motion, timestamps = batch

        if isinstance(timestamps, np.ndarray): # fixes error with unsqueeze
            timestamps = torch.from_numpy(timestamps)

        mirrored = Tensor.RandomBool() # random mirroring for data augmentation

        is_target_style = "crouch" in motion.Name.lower()

        # calculates the inverse root transformation matrix (normalizes tracking space context)
        root = Tensor.Inverse(
            motion.GetModule(RootModule).GetTransforms(timestamps, mirrored=mirrored)
        )

        frames = self.ExtractFrameFeatures(motion, timestamps, mirrored, root, is_target_style) # target frames
        
        target_frames = frames[:, :FRAME_DIM]

        # does the same based on history 
        history_timestamps = timestamps.unsqueeze(-1) + self.HistoryOffsets.to(timestamps.device)
        history_flat = history_timestamps.flatten()

        history_roots = Tensor.Inverse(motion.GetModule(RootModule).GetTransforms(history_flat, mirrored=mirrored))
        history_frames_flat = self.ExtractFrameFeatures(motion, history_flat, mirrored, history_roots, is_target_style) # history frames

        x_history_windows = history_frames_flat.reshape(len(timestamps), WINDOW_SIZE, INPUT_DIM) # reshapes history data
        
        x_history_flattened = x_history_windows.reshape(len(timestamps), WINDOW_SIZE * INPUT_DIM) # flattens data for the next layer

        return (x_history_flattened, target_frames)

    def GetEditorFeatures(self):
        features = FeedTensor("X", INPUT_DIM)
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

        cond_array = np.zeros((1, CONDITION_DIM), dtype=np.float32)
        if hasattr(self, 'StyleButton') and self.StyleButton.Active:
            cond_array[0, 1] = 1.0 
        else:
            cond_array[0, 0] = 1.0 
        features.Feed(cond_array)

        return features.GetTensor()

    # def Draw(self):
    #     with torch.no_grad():
    #         current_frame = self.GetEditorFeatures().unsqueeze(0).unsqueeze(1) # matches dimensions with yPred # [1, 1, INPUT_DIM]

    #         # deletes the first frame from the history window and adds current frame 
    #         self.EditorHistory = torch.cat([self.EditorHistory[:, 1:, :], current_frame], dim=1) 
            
    #         # Flattens the window into an array for the LSTM (dimensions expanded for condition)
    #         history_flat = self.EditorHistory.reshape(1, WINDOW_SIZE * INPUT_DIM)
            
    #         # Generates one prediction frame
    #         yPred = Tensor.ToNumPy(self.Network(history_flat))
    #         output = ReadTensor("Y", yPred)
        
    #         self.Actor.Root = self.Editor.Actor.Root # snaps the prediction actor's core location to match the master trajectory root
    #         self.Actor.SetPositions( # changes character's bone arrays based on parsed network output features
    #             Vector3.PositionFrom(output.ReadVector3(len(BONES)), self.Actor.Root)
    #         )
    #         self.Actor.SetRotations(
    #             Rotation.RotationFrom(
    #                 output.ReadRotation3D(len(BONES)), self.Actor.Root
    #             )
    #         )
    #         self.Actor.SetVelocities(
    #             Vector3.DirectionFrom(output.ReadVector3(len(BONES)), self.Actor.Root)
    #         )
    #         for bone in self.Actor.Bones:
    #             bone.RestoreLength()
    #         self.Actor.RestoreBoneAlignments()
    #         self.Actor.SyncToScene()
    #     self.Network.train()

    def Draw(self):
        with torch.no_grad():
            # 1. Read the UI Button
            is_target_style = hasattr(self, 'StyleButton') and self.StyleButton.Active
            
            # 2. Open-Loop vs Closed-Loop Switch
            if not is_target_style or not hasattr(self, 'LastRedPrediction'):
                # OPEN LOOP (Neutral): Copy the White Dummy's reality
                current_frame = self.GetEditorFeatures()
            else:
                # CLOSED LOOP (Target): Feed the Red Dummy's own last prediction back into its brain!
                current_frame = self.LastRedPrediction
                
            current_frame = current_frame.unsqueeze(0).unsqueeze(1) 

            # 3. Slide the Window
            self.EditorHistory = torch.cat([self.EditorHistory[:, 1:, :], current_frame], dim=1) 
            
            # 4. The Overwrite Fix (Force the whole window to feel the condition)
            if is_target_style:
                self.EditorHistory[:, :, FRAME_DIM] = 0.0     
                self.EditorHistory[:, :, FRAME_DIM + 1] = 1.0 
            else:
                self.EditorHistory[:, :, FRAME_DIM] = 1.0     
                self.EditorHistory[:, :, FRAME_DIM + 1] = 0.0 

            history_flat = self.EditorHistory.reshape(1, WINDOW_SIZE * INPUT_DIM)
            
            # 5. Generate Next Frame
            yPred_tensor = self.Network(history_flat)
            
            # 6. Save this prediction so we can use it in the NEXT frame (Closed-Loop)
            self.LastRedPrediction = torch.zeros(INPUT_DIM, device=yPred_tensor.device)
            self.LastRedPrediction[:FRAME_DIM] = yPred_tensor.squeeze()
            if is_target_style:
                self.LastRedPrediction[FRAME_DIM+1] = 1.0
            else:
                self.LastRedPrediction[FRAME_DIM] = 1.0

            # 7. Apply to 3D Skeleton
            yPred = Tensor.ToNumPy(yPred_tensor)
            output = ReadTensor("Y", yPred)
        
            # Cut the root string if crouching so it doesn't get dragged by the white dummy!
            if not is_target_style:
                self.Actor.Root = self.Editor.Actor.Root 
                
            self.Actor.SetPositions(Vector3.PositionFrom(output.ReadVector3(len(BONES)), self.Actor.Root))
            self.Actor.SetRotations(Rotation.RotationFrom(output.ReadRotation3D(len(BONES)), self.Actor.Root))
            self.Actor.SetVelocities(Vector3.DirectionFrom(output.ReadVector3(len(BONES)), self.Actor.Root))
            
            for bone in self.Actor.Bones:
                bone.RestoreLength()
            self.Actor.RestoreBoneAlignments()
            self.Actor.SyncToScene()
            
        self.Network.train()

def main():
    AI4Animation(Program(), mode=AI4Animation.Mode.HEADLESS)


if __name__ == "__main__":
    main()
