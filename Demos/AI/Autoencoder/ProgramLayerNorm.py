# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from ai4animation import (
    Actor,
    AI4Animation,
    #Autoencoder,
    #CosineAnnealingOptimizer,
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
    TimeSeries,
    Transform,
    Utility,
    Vector3,
)

from ai4animation.AI.Models.AutoencoderLayerNorm import EnchancedAutoencoder
#path gia data
SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/Cranberry")

sys.path.append(ASSETS_PATH)
import Definitions

EPOCH_COUNT = 150
BATCH_SIZE = 32
FRAMERATE = 30 #taxuthta kinhshs 30fps
DRAW_INTERVAL = 500 #kathe pote ananeonetai to grafhma toy loss
BONES = Definitions.FULL_BODY_NAMES
FEATURE_DIM = 12 * len(BONES) #kathe bone (sthn ousia arthrwsh) exei 12 times, koita shmwieseis 
#3 gia thesh (x,y,z), +3 gia tauthta (x,y,z) +6 gia prosanatolismo ston z kai ston y dld pou koitaei mprosta kai pou koitaei panw 
#HIDDEN_DIM = 512 #apotelesma endiamesou layer
LATENT_DIM = 256 #to latent space pou ginetai h sumpiesh


class Program:
    def Start(self):
        Utility.SetSeed(23456) #stathero seed gia na exw idia apotelesmata se kathe ektelesh, dld ksekinaw me ta idia arxikopoihmena varh se kathe ektelesh
        #na exoun idio shuffle kathe fora ta dedomena kai na exw idio augmentation sta data

        self.Dataset = Dataset(
            os.path.join(ASSETS_PATH, "Motions"),
            [#diavazei to dataset kai vriskei to root, th lekanh
                lambda x: RootModule(
                    x,
                    Definitions.HipName,
                    Definitions.LeftHipName,
                    Definitions.RightHipName,
                    Definitions.LeftShoulderName,
                    Definitions.RightShoulderName,
                    Definitions.NeckName,
                ),
                lambda x: MotionModule(x),#data augmentation, dhmiourgei kai kathreutismenes kinhseis, sthn ousia megalwnw ta data mou
                lambda x: MirrorModule(
                    x, Vector3.Axis.ZPositive, Vector3.Create(0, 0, 180)
                ),
            ],
        )

        self.DataSampler = DataSampler(#kovei to dataaset se batches
            self.Dataset,
            framerate=FRAMERATE,
            batch_size=BATCH_SIZE,
            function=self.GetTrainingFeatures,
        )

        self.Network = Tensor.ToDevice(
              EnchancedAutoencoder(
                  feature_dim=FEATURE_DIM,
                  latent_dim=LATENT_DIM,
            )
        )
        #exw provlhma gia na ftiaksw to validation set
        #error:Epoch size and batch size used in the training loop and while initializing scheduler should be the same.
        #den proxwrane me ton idio tropo kathe epoxh
        #tha valw ton optimizer afou kanw split gia na mhn exw auto to thema
        # self.Optimizer = Utility.CosineAnnealingOptimizer(#o optimizer pou exw pou allazei to lr me vash ena sunimitono (kalo gt kanw talantwseis
        #     #glitwnw provlhmata megalou kai mikroy lr)
        #     self.Network.parameters(),
        #     self.DataSampler.BatchSize,
        #     self.DataSampler.SampleCount,
        # )

        # self.LossHistory = Plotting.LossHistory(#plot to loss history
        #     "Loss History",
        #     #horizon=self.DataSampler.SampleCount,
        #     drawInterval=DRAW_INTERVAL,
        #     yScale="log",
        # )

        self.Trainer = self.Training()#ksekinaei training

    def Standalone(self):#zografizei to perivalon an exw standalone
        self.Editor = AI4Animation.Scene.AddEntity("Editor").AddComponent(
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

    def Update(self):#koitaei an stamathsw to training
        try:
            next(self.Trainer)
        except StopIteration as e:
            pass

    def Training(self):
        #spaw dedomena se 80%train kai 20%val
        print("Splitting dataset into Training (80%) and Validation (20%)...")
        all_batches = list(self.DataSampler.SampleBatchesWithinMotions(1, EPOCH_COUNT))
        
        total_batches = len(all_batches)
        split_idx = int(0.8 * total_batches) #to 80% twn batch pane gia ekpaideush
        #vazw edw ton optimizer
        print("Splitting dataset into Training (80%) and Validation (20%)...")
        all_batches = list(self.DataSampler.SampleBatchesWithinMotions(1, EPOCH_COUNT))
        
        total_batches = len(all_batches)
        split_idx = int(0.8 * total_batches) # το 80% των batch πάνε για εκπαίδευση
        
        train_batches = all_batches[:split_idx]
        val_batches = all_batches[split_idx:]
        
        #upologizw sunoliko athroisma twn deigmatwn se ola ta training batches gia na to dwsw ston optimizer
        #kai na mhn pathainei tso tso
        total_train_samples = sum([batch.shape[0] for batch in train_batches])
        
        self.Optimizer = Utility.CosineAnnealingOptimizer(
            self.Network.parameters(),
            self.DataSampler.BatchSize,
            total_train_samples
        )
        
        #grapse poia batch einai gia ti
        print(f"Total batches: {total_batches} | Train: {len(train_batches)} | Val: {len(val_batches)}")

        #listes gia apothikeush twn loss
        train_losses_history = []
        val_losses_history = []
        best_val_loss = float('inf')
        
        #to training loop + to validation 
        for epoch in range(1, EPOCH_COUNT + 1):
            print(f"\n--- Epoch {epoch}/{EPOCH_COUNT} ---")
            
            #fash ekpaideushs
            self.Network.train()
            epoch_train_loss = 0.0
            
            for i, batch in enumerate(train_batches): # Προσθέσαμε το i και το enumerate
                _, loss = self.Network.learn(batch, epoch == 1)
                
                if isinstance(loss, dict):
                    tensor_loss = sum(loss.values())
                else:
                    tensor_loss = loss
                
                self.Optimizer.Update(batch.shape[0], tensor_loss) 
                epoch_train_loss += tensor_loss.item()
                #print mpares
                #progress = 100 * (i + 1) / len(train_batches)
                #print(f"Training Progress: {progress:.1f}%", end="\r")
                
                yield
            print(" " * 50, end="\r")
                
            avg_train_loss = epoch_train_loss / len(train_batches)
            train_losses_history.append(avg_train_loss)
            
            #VALIDATIONNN
            self.Network.eval() #valto se mode evaluation
            epoch_val_loss = 0.0
            #profanws den upologizoume ta grands sto validation
            with torch.no_grad(): 
                for batch in val_batches:
                    #pare to ekastote batch kai kanto tensora
                    inputs_tensor = Tensor.ToDevice(batch.clone().detach().float())

                    #GIA LAYER NORM AUTOENCODER
                    #kane normalize ta inputs kai perna apo autoencoder
                    norm_inputs = self.Network.Statistics.Normalize(inputs_tensor)
                    latent = self.Network.Encoder(norm_inputs)
                    norm_preds = self.Network.Decoder(latent)
                    #sugkrine ta kanonikopoihmena dedomena
                    loss = torch.nn.functional.mse_loss(norm_preds, norm_inputs).item()

                    
                    epoch_val_loss += loss
                    yield
            #vres validation loss
            avg_val_loss = epoch_val_loss / len(val_batches)
            val_losses_history.append(avg_val_loss)
            
            print(f"Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
            
            #otan vriskei kalutero validation loss kanto ena print
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(">>> New best validation loss! <<<")
                
            #self.LossHistory.Print()

            #kalese sunarthsh gia na kaneis plot ta loss
            self.PlotTrainVal(train_losses_history, val_losses_history, epoch)
        #auta einai gia to plot
        plt.ioff() 
        plt.savefig("loss_history_NORM.png", dpi=300, bbox_inches='tight') 
        plt.show() 

        torch.save(self.Network.state_dict(), "layernorm_weights.pth")

    def PlotTrainVal(self, train_losses, val_losses, epoch):
        #gia na kanei plot tis grafikes
        plt.ion() 
        plt.clf()
        
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue', linewidth=2)
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='orange', linewidth=2)
        
        plt.title(f'Enchanced Autoencoder MSE Loss (Epoch {epoch}/{EPOCH_COUNT})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Log Scale)')
        plt.yscale('log') #log klimaka gt to loss peftei apotoma sthn arxh 
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        
        plt.pause(0.01) #pasuh gia na ananewsei to parathuro

    def GetTrainingFeatures(self, batch):
        motion, timestamps = batch#me auta travaei dedomena
        mirrored = Tensor.RandomBool()

        inputs = FeedTensor("X", (len(timestamps), FEATURE_DIM)) #edw dhmiourgw tensora 32x600 kai tha dwsw ta dedomena
        #vrhskw inverse tou m/s ths rizas dld ths lekanhs gt thelw to diktuo
        #na mathainei topikes suntetagmenes (se sxesh me th lekane) kai oxi global, dld se olo to xarth
        #san na kanonikopoiw ws pros (0,0,0) ths lekanhs
        window = Tensor.RandomUniform(min=0.0, max=1.0)
        smoothing = TimeSeries(-window / 2, window / 2, 10)
        rootInv = Tensor.Inverse(
            motion.GetModule(RootModule).GetTransforms(
                timestamps, mirrored=mirrored, smoothing=smoothing
            )
        )

        #upologizw theseis kai taxuthtes ostwn se sxesh me th lekane
        transforms = Transform.TransformationFrom(
            motion.GetBoneTransformations(timestamps, BONES, mirrored=mirrored),
            rootInv.reshape(-1, 1, 4, 4),
        )
        velocities = Vector3.DirectionFrom(
            motion.GetBoneVelocities(timestamps, BONES, mirrored=mirrored),
            rootInv.reshape(-1, 1, 4, 4),
        )#dinw dedomena sto feed, ston 32x600 tensora pou eftiaksa sthn arxh
        inputs.Feed(Transform.GetPosition(transforms))
        inputs.Feed(Transform.GetAxisZ(transforms))
        inputs.Feed(Transform.GetAxisY(transforms))
        inputs.Feed(velocities)
        #epistrefei teliko array me 32kare x 600diastaseis
        return inputs.GetTensor()

    def GetEditorFeatures(self):
        #kanei to idio pragma me thn prohgoumenh sunarthsh alla trexei
        #60fores/sec otan exw anoixto to 3d parathuro me ton actor
        features = FeedTensor("X", FEATURE_DIM) #tensoras 1x600, pairnei mono ena kare
        #diavazei to 3d montelo pou vrhsketai kai pairnei ta dedomena toy, th lekanh ekeinh th xronikh stigmh
        #kai kanei ola tou ta osta suntetagmenes
        root = self.Editor.Actor.Root
        transforms = Transform.TransformationTo(
            self.Editor.Actor.GetTransforms(BONES), root
        )
        #kanei to idio me panw
        velocities = Vector3.DirectionTo(self.Editor.Actor.GetVelocities(BONES), root)
        features.Feed(Transform.GetPosition(transforms))
        features.Feed(Transform.GetAxisZ(transforms))
        features.Feed(Transform.GetAxisY(transforms))
        features.Feed(velocities)
        return features.GetTensor()

    def Draw(self):#zwgrafaei to 3d
        self.Network.eval()
        with torch.no_grad():
            xBatch = self.GetEditorFeatures()
            yPred = Tensor.ToNumPy(self.Network(xBatch))
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
    AI4Animation(Program(), mode=AI4Animation.Mode.HEADLESS)


if __name__ == "__main__":
    main()
