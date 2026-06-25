# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# GCN with Learnable Adjacency Matrix for Human Motion Prediction
# Based on: Kipf & Welling (2016) as reviewed in Subbulakshmi et al. (2025)
#
# Key upgrade over basic GCN:
#   The adjacency matrix A is initialised from the skeleton structure
#   but then made learnable — the model discovers which joint relationships
#   actually matter for prediction (e.g. left foot ↔ right foot coordination
#   even without a direct skeletal link).
#
# GCN operation: M(l+1) = σ(A_learned · M(l) · W(l))

import torch
import torch.nn as nn
import torch.nn.functional as F
from ai4animation.AI import Losses, Stats
from ai4animation.AI.Library import Defaults


def build_skeleton_adjacency(num_joints=27):
    """Skeleton adjacency for Cranberry 27-bone rig."""
    edges = [
        (0,1),(0,5),(0,9),        # Hip → legs + spine
        (1,2),(2,3),(3,4),        # Left leg
        (5,6),(6,7),(7,8),        # Right leg
        (9,10),(10,11),(11,12),   # Spine
        (12,13),(13,14),          # Neck → Head
        (12,15),(15,16),(16,17),(17,18),(18,19),(19,20),  # Left arm
        (12,21),(21,22),(22,23),(23,24),(24,25),(25,26),  # Right arm
    ]
    A = torch.zeros(num_joints, num_joints)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A = A + torch.eye(num_joints)
    D = A.sum(dim=1)
    D_inv_sqrt = torch.diag(D.pow(-0.5))
    return D_inv_sqrt @ A @ D_inv_sqrt


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=Defaults.Dropout):
        super().__init__()
        self.Linear  = nn.Linear(in_features, out_features, bias=True)
        self.Norm    = nn.LayerNorm(out_features)
        self.Dropout = nn.Dropout(dropout)

    def forward(self, x, A):
        """
        x: [Batch, NumJoints, in_features]
        A: [NumJoints, NumJoints]  (learnable, softmax-normalised)
        """
        x = torch.bmm(A.unsqueeze(0).expand(x.size(0), -1, -1), x)
        x = self.Linear(x)
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
        self.FrameDim      = num_joints * feat_per_joint

        self.InputStats  = Stats.RunningStats(self.FrameDim)
        self.OutputStats = Stats.RunningStats(self.FrameDim)

        # Learnable adjacency — initialised from skeleton, then optimised
        # Stored as raw logits, softmax-normalised per row during forward pass
        # This lets the model learn which joint relationships matter most
        A_init = build_skeleton_adjacency(num_joints)
        self.AdjLogits = nn.Parameter(A_init.clone())

        self.InputProjection = nn.Linear(feat_per_joint, hidden_dim)

        self.GCNLayers = nn.ModuleList([
            GCNLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(num_gcn_layers)
        ])

        self.OutputProjection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feat_per_joint),
        )

    def _get_adjacency(self):
        """
        Normalise learnable logits row-wise with softmax.
        This keeps the adjacency well-behaved (rows sum to 1)
        while allowing gradients to update which joints attend to which.
        """
        return F.softmax(self.AdjLogits, dim=-1)

    def _reshape_to_graph(self, x_flat, batch_size):
        return x_flat.reshape(batch_size, self.NumJoints, self.FeatPerJoint)

    def _reshape_to_flat(self, x_graph):
        return x_graph.reshape(x_graph.size(0), self.FrameDim)

    def _forward_gcn(self, x_flat):
        B = x_flat.size(0)
        A = self._get_adjacency()               # [J, J] normalised

        x = self._reshape_to_graph(x_flat, B)   # [B, J, 12]
        x = F.elu(self.InputProjection(x))       # [B, J, hidden]

        for layer in self.GCNLayers:
            x = layer(x, A)                      # [B, J, hidden]

        x = self.OutputProjection(x)             # [B, J, 12]
        return self._reshape_to_flat(x)          # [B, FrameDim]

    def forward(self, history, generate_steps=1):
        """
        history: [Batch, WindowSize, FrameDim]
        returns: [Batch, generate_steps, FrameDim]
        """
        batch_size = history.shape[0]
        generated  = []
        current    = history.clone()

        for _ in range(generate_steps):
            x_last     = current[:, -1, :]
            x_norm     = self.InputStats.Normalize(x_last)
            pred_norm  = self._forward_gcn(x_norm)
            next_frame = self.OutputStats.Denormalize(pred_norm)

            generated.append(next_frame)
            current = torch.cat(
                [current[:, 1:, :], next_frame.unsqueeze(1)], dim=1
            )

        return torch.stack(generated, dim=1)

    def learn(self, input_windows, target_frames, update_stats):
        """
        input_windows: [Batch, WindowSize, FrameDim]
        target_frames: [Batch, FrameDim]
        """
        if update_stats:
            self.InputStats.Update(input_windows.reshape(-1, self.FrameDim))
            self.OutputStats.Update(target_frames)

        x_last = input_windows[:, -1, :]
        x_norm = self.InputStats.Normalize(x_last)
        y_norm = self.OutputStats.Normalize(target_frames)

        pred   = self._forward_gcn(x_norm)
        loss   = Losses.MSE(pred, y_norm)
        return {"MSE Loss": loss}
