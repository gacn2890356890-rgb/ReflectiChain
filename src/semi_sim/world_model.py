"""
Supply Chain World Model (SC-WM).
Generative latent model for simulating supply chain state trajectories.
Implements JEPA-inspired architecture for topology-aware risk propagation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from config import WorldModelConfig, SemiSimConfig


class SCWorldModel(nn.Module):
    """
    Supply Chain World Model: learns latent dynamics of supply chain states.

    Architecture:
        - Encoder: maps (node_states, graph_structure) -> latent z
        - Dynamics Model: z -> z' via graph-aware transition
        - Decoder: z' -> predicted_reward, predicted_next_state
    """

    def __init__(self, cfg: WorldModelConfig, semi_cfg: SemiSimConfig):
        super().__init__()
        self.cfg = cfg
        self.semi_cfg = semi_cfg

        self.state_dim = 5      # [inventory, cash, compliance, risk, upstream_risk] per node
        self.node_dim = self.state_dim * semi_cfg.num_nodes  # 5 * 6 = 30
        self.latent_dim = cfg.latent_dim  # 64
        self.hidden_dim = cfg.hidden_dim  # 128

        # ── Encoder: node features + graph topology -> latent ──────────────────
        self.encoder = nn.Sequential(
            nn.Linear(self.node_dim + semi_cfg.num_edges * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )

        # ── Dynamics Model: latent transition ─────────────────────────────────
        self.dynamics_layers = nn.ModuleList()
        for _ in range(cfg.num_layers):
            self.dynamics_layers.append(nn.Sequential(
                nn.Linear(self.latent_dim + 4, self.hidden_dim),  # +4 for action encoding
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            ))
        self.dynamics_out = nn.Linear(self.hidden_dim, self.latent_dim)
        self.dynamics_residual = nn.Linear(self.latent_dim, self.latent_dim)

        # ── Decoder: latent -> reward prediction ───────────────────────────────
        self.reward_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Tanh(),  # Reward in [-1, 1]
        )

        # ── Decoder: latent -> state delta prediction ─────────────────────────
        self.state_delta_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.node_dim),
        )

        # ── Graph convolution for topology-aware dynamics ──────────────────────
        self.edge_encoder = nn.Linear(2, 8)  # [weight, flow] -> 8-dim
        self.graph_agg = nn.MultiheadAttention(
            embed_dim=self.latent_dim, num_heads=4, batch_first=True
        )

    def encode(self, node_features: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Encode node states and graph structure into latent representation.

        Args:
            node_features: (batch, num_nodes * state_dim)
            edge_features: (batch, num_edges * 2)  [weight, current_flow per edge]

        Returns:
            z: (batch, latent_dim)
        """
        # Encode edge features
        edge_emb = self.edge_encoder(edge_features)  # (batch, num_edges, 8)
        edge_pooled = edge_emb.mean(dim=1)           # (batch, 8)

        # Concatenate node features + edge summary
        combined = torch.cat([node_features, edge_pooled], dim=-1)  # (batch, node_dim+8)
        z = self.encoder(combined)
        return z

    def forward_dynamics(
        self,
        z: torch.Tensor,
        action_encoding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply latent dynamics transition.

        Args:
            z: (batch, latent_dim) current latent state
            action_encoding: (batch, 4) [procurement, shipment, compliance_shift, target_compliance]

        Returns:
            z_next: (batch, latent_dim) predicted next latent state
        """
        x = torch.cat([z, action_encoding], dim=-1)
        for layer in self.dynamics_layers:
            x = layer(x)
        delta_z = self.dynamics_out(x)
        z_next = F.gelu(z + self.dynamics_residual(delta_z))
        return z_next

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode latent state to reward and state delta.

        Returns:
            reward: (batch, 1) predicted physical execution reward
            state_delta: (batch, node_dim) predicted state change
        """
        reward = self.reward_head(z)
        state_delta = self.state_delta_head(z)
        return reward, state_delta

    def predict(
        self,
        node_features: np.ndarray,
        edge_features: np.ndarray,
        action_encoding: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Main inference API: predict next state reward and delta.

        Args:
            node_features: (num_nodes * state_dim,) numpy array
            edge_features: (num_edges * 2,) numpy array
            action_encoding: (4,) numpy array

        Returns:
            predicted_reward: float in [-1, 1]
            predicted_state_delta: numpy array
        """
        self.eval()
        with torch.no_grad():
            node_t = torch.FloatTensor(node_features).unsqueeze(0)
            edge_t = torch.FloatTensor(edge_features).unsqueeze(0)
            action_t = torch.FloatTensor(action_encoding).unsqueeze(0)

            z = self.encode(node_t, edge_t)
            z_next = self.forward_dynamics(z, action_t)
            reward, state_delta = self.decode(z_next)

            return reward.item(), state_delta.squeeze(0).numpy()


class WorldModelWrapper:
    """
    Wrapper around SC-WM that handles:
    - Trajectory rollout (multiple steps into the future)
    - Latent rehearsal for candidate action evaluation
    """

    def __init__(self, cfg: WorldModelConfig, semi_cfg: SemiSimConfig, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.wm = SCWorldModel(cfg, semi_cfg).to(device)
        self.optimizer = torch.optim.AdamW(
            self.wm.parameters(), lr=3e-4, weight_decay=0.01
        )
        self.wm.train()

    def latent_rollout(
        self,
        current_node_features: np.ndarray,
        current_edge_features: np.ndarray,
        action_encoding: np.ndarray,
        num_steps: int = 5,
    ) -> Tuple[float, List[np.ndarray]]:
        """
        Perform multi-step latent trajectory rehearsal.

        Returns:
            final_reward: accumulated reward along trajectory
            trajectory_states: list of latent states
        """
        self.wm.eval()
        with torch.no_grad():
            node_t = torch.FloatTensor(current_node_features).unsqueeze(0).to(self.device)
            edge_t = torch.FloatTensor(current_edge_features).unsqueeze(0).to(self.device)
            action_t = torch.FloatTensor(action_encoding).unsqueeze(0).to(self.device)

            z = self.wm.encode(node_t, edge_t)
            trajectory = [z.cpu().numpy()]

            accumulated_reward = 0.0
            for step in range(num_steps):
                z = self.wm.forward_dynamics(z, action_t)
                reward, _ = self.wm.decode(z)
                accumulated_reward += reward.item()
                trajectory.append(z.cpu().numpy())

        # Return average reward per step
        avg_reward = accumulated_reward / num_steps
        return avg_reward, trajectory

    def evaluate_candidate(
        self,
        env_state,
        action,
    ) -> float:
        """
        Evaluate a candidate action using the world model.

        Returns:
            predicted_reward: WM-based physical grounding score
        """
        node_features = np.concatenate([s.to_vector() for s in env_state.node_states.values()])

        # Build edge features
        edge_features = []
        for u, v in env_state.G.edges():
            w = env_state.G.edges[u, v].get('weight', 0.3)
            flow = env_state.G.edges[u, v].get('current_flow', 0)
            edge_features.extend([w, flow])
        edge_features = np.array(edge_features, dtype=np.float32)

        action_encoding = np.array([
            action.procurement_volume / 10000.0,
            action.shipment_volume / 10000.0,
            action.compliance_shift / 100.0,
            action.target_compliance / 100.0,
        ], dtype=np.float32)

        predicted_reward, _ = self.predict(node_features, edge_features, action_encoding)
        return predicted_reward

    def update(
        self,
        node_features: np.ndarray,
        edge_features: np.ndarray,
        action_encoding: np.ndarray,
        observed_reward: float,
        observed_state_delta: np.ndarray,
    ) -> float:
        """
        Update the world model from observed transitions.
        Uses MSE loss on reward and state delta prediction.
        """
        self.wm.train()
        node_t = torch.FloatTensor(node_features).unsqueeze(0).to(self.device)
        edge_t = torch.FloatTensor(edge_features).unsqueeze(0).to(self.device)
        action_t = torch.FloatTensor(action_encoding).unsqueeze(0).to(self.device)
        reward_t = torch.FloatTensor([[observed_reward]]).to(self.device)
        delta_t = torch.FloatTensor(observed_state_delta).unsqueeze(0).to(self.device)

        z = self.wm.encode(node_t, edge_t)
        z_next = self.wm.forward_dynamics(z, action_t)
        pred_reward, pred_delta = self.wm.decode(z_next)

        reward_loss = F.mse_loss(pred_reward, reward_t)
        delta_loss = F.mse_loss(pred_delta, delta_t)
        loss = reward_loss + 0.5 * delta_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def save(self, path: str):
        torch.save(self.wm.state_dict(), path)

    def load(self, path: str):
        self.wm.load_state_dict(torch.load(path, map_location=self.device))
        self.wm.eval()

    def predict(
        self,
        node_features: np.ndarray,
        edge_features: np.ndarray,
        action_encoding: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        return self.wm.predict(node_features, edge_features, action_encoding)
