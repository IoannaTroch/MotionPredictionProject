# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Basic Graph Convolutional Network for Human Motion Prediction
# Based on: Kipf & Welling (2016) - Semi-supervised classification with GCNs
# As reviewed in: Subbulakshmi et al. (2025) - GCNN on Human Motion Prediction
#
# Architecture:
#   Input: [Batch, WindowSize, NumJoints, FeaturesPerJoint]
#   GCN layers: M(l+1) = σ(A · M(l) · W(l))
#   Where A = normalised skeleton adjacency matrix (which joints connect)
#         M = joint feature matrix
#         W = learnable weight matrix
#   Output: [Batch, NumJoints * FeaturesPerJoint] — next frame prediction
#
# Key difference from MLP:
#   MLP treats all 324 features as a flat vector, ignoring joint relationships.
#   GCN explicitly models which joints are connected in the skeleton graph,
#   so it learns that e.g. knee movement depends on hip, not on wrist.

import torch
import torch.nn as nn
import torch.nn.functional as F
from ai4animation.AI import Losses, Stats
from ai4animation.AI.Library import Defaults


def build_skeleton_adjacency(num_joints=27):
    """
    Build the skeleton adjacency matrix for the Cranberry character.
    Joints (FULL_BODY_NAMES index):
      0  b_root        (Hip)
      1  b_l_upleg     (LeftHip)
      2  b_l_leg       (LeftKnee)
      3  b_l_talocrural(LeftAnkle)
      4  b_l_ball      (LeftBall)
      5  b_r_upleg     (RightHip)
      6  b_r_leg       (RightKnee)
      7  b_r_talocrural(RightAnkle)
      8  b_r_ball      (RightBall)
      9  b_spine0
     10  b_spine1
     11  b_spine2
     12  b_spine3
     13  b_neck0       (Neck)
     14  b_head        (Head)
     15  b_l_shoulder  (LeftScap)
     16  p_l_scap      (LeftShoulder)
     17  b_l_arm       (LeftArm)
     18  b_l_forearm   (LeftElbow)
     19  b_l_wrist_twist
     20  b_l_wrist     (LeftWrist)
     21  b_r_shoulder  (RightScap)
     22  p_r_scap      (RightShoulder)
     23  b_r_arm       (RightArm)
     24  b_r_forearm   (RightElbow)
     25  b_r_wrist_twist
     26  b_r_wrist     (RightWrist)
    """
    edges = [
        # Spine chain
        (0, 1),   # Hip → LeftHip
        (0, 5),   # Hip → RightHip
        (0, 9),   # Hip → Spine0
        # Left leg
        (1, 2),   # LeftHip → LeftKnee
        (2, 3),   # LeftKnee → LeftAnkle
        (3, 4),   # LeftAnkle → LeftBall
        # Right leg
        (5, 6),   # RightHip → RightKnee
        (6, 7),   # RightKnee → RightAnkle
        (7, 8),   # RightAnkle → RightBall
        # Spine
        (9, 10),  # Spine0 → Spine1
        (10, 11), # Spine1 → Spine2
        (11, 12), # Spine2 → Spine3
        (12, 13), # Spine3 → Neck
        (13, 14), # Neck → Head
        # Left arm
        (12, 15), # Spine3 → LeftScap
        (15, 16), # LeftScap → LeftShoulder
        (16, 17), # LeftShoulder → LeftArm
        (17, 18), # LeftArm → LeftElbow
        (18, 19), # LeftElbow → LeftWristTwist
        (19, 20), # LeftWristTwist → LeftWrist
        # Right arm
        (12, 21), # Spine3 → RightScap
        (21, 22), # RightScap → RightShoulder
        (22, 23), # RightShoulder → RightArm
        (23, 24), # RightArm → RightElbow
        (24, 25), # RightElbow → RightWristTwist
        (25, 26), # RightWristTwist → RightWrist
    ]

    # Build symmetric adjacency matrix
    A = torch.zeros(num_joints, num_joints)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0

    # Add self-loops (each joint connects to itself)
    A = A + torch.eye(num_joints)

    # Normalise: D^{-1/2} A D^{-1/2}  (symmetric normalisation from Kipf & Welling)
    D = A.sum(dim=1)                          # degree vector
    D_inv_sqrt = torch.diag(D.pow(-0.5))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt      # [NumJoints, NumJoints]

    return A_norm


class GCNLayer(nn.Module):
    """
    Single GCN layer: M(l+1) = σ(A · M(l) · W(l))
    A is fixed (skeleton structure), W is learnable.
    """
    def __init__(self, in_features, out_features, dropout=Defaults.Dropout):
        super().__init__()
        self.Linear   = nn.Linear(in_features, out_features, bias=True)
        self.Norm     = nn.LayerNorm(out_features)
        self.Dropout  = nn.Dropout(dropout)

    def forward(self, x, A):
        """
        x: [Batch, NumJoints, in_features]
        A: [NumJoints, NumJoints]
        returns: [Batch, NumJoints, out_features]
        """
        # Graph convolution: aggregate neighbour features via adjacency
        x = torch.bmm(A.unsqueeze(0).expand(x.size(0), -1, -1), x)  # [B, J, F]
        x = self.Linear(x)    # learnable projection
        x = self.Norm(x)
        x = F.elu(x)
        x = self.Dropout(x)
        return x


class Model(nn.Module):
    def __init__(
        self,
        num_joints,
        feat_per_joint,
        window_size,
        hidden_dim=128,
        num_gcn_layers=4,
        dropout=Defaults.Dropout,
    ):
        super().__init__()

        self.NumJoints     = num_joints
        self.FeatPerJoint  = feat_per_joint
        self.WindowSize    = window_size
        self.FrameDim      = num_joints * feat_per_joint  # flat frame size

        # Input/output running statistics operate on flat frames
        self.InputStats  = Stats.RunningStats(self.FrameDim)
        self.OutputStats = Stats.RunningStats(self.FrameDim)

        # Fixed skeleton adjacency matrix — registered as buffer so it moves
        # to GPU with the model automatically
        A = build_skeleton_adjacency(num_joints)
        self.register_buffer('A', A)

        # Project from feat_per_joint to hidden_dim at input
        self.InputProjection = nn.Linear(feat_per_joint, hidden_dim)

        # Stack of GCN layers — each operates on [Batch, NumJoints, hidden_dim]
        self.GCNLayers = nn.ModuleList([
            GCNLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(num_gcn_layers)
        ])

        # After GCN, aggregate across joints and project to output frame
        # Global average pooling over joints then linear to frame_dim
        self.OutputProjection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feat_per_joint),
        )

    def _reshape_to_graph(self, x_flat, batch_size):
        """Flat [B, FrameDim] → graph [B, NumJoints, FeatPerJoint]"""
        return x_flat.reshape(batch_size, self.NumJoints, self.FeatPerJoint)

    def _reshape_to_flat(self, x_graph):
        """Graph [B, NumJoints, FeatPerJoint] → flat [B, FrameDim]"""
        return x_graph.reshape(x_graph.size(0), self.FrameDim)

    def _forward_gcn(self, x_flat):
        """
        Run one forward pass through GCN stack.
        x_flat: [Batch, FrameDim] — normalised flat frame
        returns: [Batch, FrameDim] — predicted next frame (normalised)
        """
        B = x_flat.size(0)

        # Reshape to graph structure
        x = self._reshape_to_graph(x_flat, B)  # [B, J, 12]

        # Project features to hidden dim
        x = self.InputProjection(x)             # [B, J, hidden]
        x = F.elu(x)

        # GCN layers — graph convolution over skeleton
        for layer in self.GCNLayers:
            x = layer(x, self.A)                # [B, J, hidden]

        # Per-joint output projection back to feat_per_joint
        x = self.OutputProjection(x)            # [B, J, 12]

        # Flatten back to frame
        return self._reshape_to_flat(x)         # [B, FrameDim]

    def forward(self, history, generate_steps=1):
        """
        history:        [Batch, WindowSize, FrameDim]
        generate_steps: how many future frames to produce autoregressively
        returns:        [Batch, generate_steps, FrameDim]
        """
        batch_size = history.shape[0]
        generated  = []
        current    = history.clone()

        for _ in range(generate_steps):
            # Use the most recent frame as input (last in window)
            x_last = current[:, -1, :]                         # [B, FrameDim]
            x_norm = self.InputStats.Normalize(x_last)

            pred_norm  = self._forward_gcn(x_norm)             # [B, FrameDim]
            next_frame = self.OutputStats.Denormalize(pred_norm)

            generated.append(next_frame)

            # Slide window
            current = torch.cat(
                [current[:, 1:, :], next_frame.unsqueeze(1)], dim=1
            )

        return torch.stack(generated, dim=1)                   # [B, steps, FrameDim]

    def learn(self, input_windows, target_frames, update_stats):
        """
        input_windows:  [Batch, WindowSize, FrameDim]
        target_frames:  [Batch, FrameDim]
        """
        if update_stats:
            flat_inputs = input_windows.reshape(-1, self.FrameDim)
            self.InputStats.Update(flat_inputs)
            self.OutputStats.Update(target_frames)

        # Use the most recent frame in each window as input
        x_last = input_windows[:, -1, :]                       # [B, FrameDim]
        x_norm = self.InputStats.Normalize(x_last)
        y_norm = self.OutputStats.Normalize(target_frames)

        pred   = self._forward_gcn(x_norm)

        loss = Losses.MSE(pred, y_norm)
        return {"MSE Loss": loss}
