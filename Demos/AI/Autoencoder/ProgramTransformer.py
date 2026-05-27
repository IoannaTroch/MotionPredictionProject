#normal MyProgram
# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
from pathlib import Path

import torch
from ai4animation import (
    Actor,
    AI4Animation,
    Autoencoder,
    CosineAnnealingOptimizer,
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

#from ai4animation.AI.Models.MyAutoencoder import MyAdvancedAutoencoder
from ai4animation.AI.Models.Transformer import TransformerAutoencoder

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

        

        # self.Network = Tensor.ToDevice(
        #      MyAdvancedAutoencoder(
        #          feature_dim=FEATURE_DIM,
        #          latent_dim=LATENT_DIM,
        #     )
        #  )
        
        self.Network = Tensor.ToDevice(
            TransformerAutoencoder(
                feature_dim=FEATURE_DIM,
                latent_dim=LATENT_DIM,
            )
        )

        self.Optimizer = CosineAnnealingOptimizer(#o optimizer pou exw pou allazei to lr me vash ena sunimitono (kalo gt kanw talantwseis
            #glitwnw provlhmata megalou kai mikroy lr)
            self.Network.parameters(),
            self.DataSampler.BatchSize,
            self.DataSampler.BatchCount,
        )

        self.LossHistory = Plotting.LossHistory(#plot to loss history
            "Loss History",
            horizon=self.DataSampler.BatchCount,
            drawInterval=DRAW_INTERVAL,
            yScale="log",
        )

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
        #to klasiko training loop
        for epoch in range(1, EPOCH_COUNT + 1):
            print("Epoch", epoch)
            for batch in self.DataSampler.SampleBatchesWithinMotions(
                epoch, EPOCH_COUNT
            ):
                _, loss = self.Network.learn(batch, epoch == 1)
                self.Optimizer.Update(loss) 
                self.LossHistory.Add(loss)
                yield
            self.LossHistory.Print()

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

    def Draw(self):#zwgrafaei
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
