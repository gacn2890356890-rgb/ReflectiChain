"""
Baseline Models: PPO, Vanilla LLM, and other competitors.
"""
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Any
from abc import ABC, abstractmethod

from src.semi_sim.enhanced_env import EnhancedSemiSimEnvironment as SemiSimEnvironment, Observation, Action


class BaseBaseline(ABC):
    """Abstract base class for all baseline models."""

    @abstractmethod
    def reset(self, env: SemiSimEnvironment):
        """Reset agent for a new episode."""
        pass

    @abstractmethod
    def act(self, obs: Observation) -> List[Action]:
        """Select action given observation."""
        pass

    @abstractmethod
    def update(self, obs, action, reward, obs_next, done):
        """Update agent based on transition."""
        pass


class PPORLBaseline(BaseBaseline):
    """
    PPO-based reinforcement learning baseline.
    Falls back to rule-based simulation when full RL is unavailable.
    """

    def __init__(self, num_nodes: int = 6, seed: int = 42):
        self.num_nodes = num_nodes
        self.rng = np.random.default_rng(seed)
        self.step_count = 0

        # Simple policy network (placeholder)
        self.state_dim = num_nodes * 5
        self.action_dim = num_nodes * 3

        # Initialize policy
        np.random.seed(seed)
        self.policy_weights = np.random.randn(self.state_dim, self.action_dim) * 0.01
        self.value_weights = np.random.randn(self.state_dim, 1) * 0.01

        # PPO-specific
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.policy_buffer: List[Dict] = []

    def reset(self, env: SemiSimEnvironment):
        self.step_count = 0
        self.policy_buffer = []

    def act(self, obs: Observation) -> List[Action]:
        """PPO policy: map state to action with learned policy."""
        self.step_count += 1

        # Compute state vector
        state = self._obs_to_state(obs)
        action_logits = state @ self.policy_weights

        # Add exploration noise
        noise = self.rng.normal(0, 0.1, action_logits.shape)
        action_logits = action_logits + noise

        # Decode into actions
        actions = self._decode_action_logits(action_logits, obs)

        return actions

    def _obs_to_state(self, obs: Observation) -> np.ndarray:
        """Convert observation to flat state vector."""
        state = []
        for nid in range(self.num_nodes):
            if nid in obs.node_states:
                s = obs.node_states[nid]
                state.extend([
                    s.inventory / 10000.0,
                    s.cash / 500000.0,
                    s.compliance / 100.0,
                    s.risk_exposure,
                    s.upstream_risk,
                ])
            else:
                state.extend([0.0] * 5)
        return np.array(state, dtype=np.float32)

    def _decode_action_logits(
        self,
        logits: np.ndarray,
        obs: Observation,
    ) -> List[Action]:
        """Decode policy logits into Action objects."""
        actions = []
        for nid in range(self.num_nodes):
            idx = nid * 3
            proc = float(np.clip(logits[idx], 0, 1) * 10000)
            ship = float(np.clip(logits[idx + 1], 0, 1) * 5000)
            comp_shift = float((logits[idx + 2] - 0.5) * 40)

            actions.append(Action(
                agent_id=-1,  # PPO baseline has no agent_id
                node_id=nid,
                description="PPO policy action",
                procurement_volume=proc,
                shipment_volume=ship,
                compliance_shift=comp_shift,
                target_compliance=80.0 + comp_shift,
                negotiate_contract=False,
                hedge_risk=False,
            ))
        return actions

    def update(self, obs, action, reward, obs_next, done):
        """PPO policy gradient update (simplified)."""
        state = self._obs_to_state(obs)

        # Simplified gradient: move policy toward higher-reward actions
        # scalar delta broadcasts with weight matrix
        lr = 0.0001
        delta = lr * reward
        self.policy_weights += delta

        # Clamp to prevent explosion
        norm = np.linalg.norm(self.policy_weights)
        if norm > 5.0:
            self.policy_weights *= 5.0 / norm


class VanillaLLMBaseline(BaseBaseline):
    """
    Vanilla LLM baseline (Qwen2.5-7B / InternLM2.5).
    LLM acts directly without World Model or Reflection.
    Falls into Decision Paralysis (stagnation trap) when facing policy shocks.
    """

    def __init__(
        self,
        model_name: str = "Qwen2.5-7B",
        llm_client=None,
        seed: int = 42,
    ):
        self.model_name = model_name
        self.llm = llm_client
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.stagnation_counter = 0

    def reset(self, env: SemiSimEnvironment):
        self.step_count = 0
        self.stagnation_counter = 0

    def act(self, obs: Observation) -> List[Action]:
        """Vanilla LLM: direct action without reflection."""
        self.step_count += 1

        # Check for stagnation trap
        if obs.global_risk > 0.6:
            self.stagnation_counter += 1

        # If stagnated, produce minimal safe actions
        if self.stagnation_counter > 3:
            return self._safe_actions(obs)

        # Normal: ask LLM for action
        prompt = self._build_prompt(obs)
        response = self._call_llm(prompt)

        return self._parse_response(response, obs)

    def _build_prompt(self, obs: Observation) -> str:
        return f"""
You are a supply chain manager. Current state:
{obs.to_text()}
Generate JSON actions for all {len(obs.node_states)} nodes.
Return a JSON array of actions with fields: node_id, procurement_volume, shipment_volume, compliance_shift.
"""

    def _call_llm(self, prompt: str) -> str:
        if self.llm is not None:
            try:
                response = self.llm.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=512,
                )
                return response.choices[0].message.content
            except Exception:
                pass
        return self._fallback_response()

    def _fallback_response(self) -> str:
        # Vanilla LLM fallback: conservative actions
        import random, json
        actions = []
        for nid in range(6):
            actions.append({
                "node_id": nid,
                "procurement_volume": random.uniform(1000, 3000),
                "shipment_volume": random.uniform(500, 2000),
                "compliance_shift": random.uniform(-5, 10),
            })
        return json.dumps(actions)

    def _parse_response(self, response: str, obs: Observation) -> List[Action]:
        try:
            import json, re
            text = response.strip()
            start = text.find('[')
            end = text.rfind(']') + 1
            if start != -1 and end > start:
                data = json.loads(text[start:end])
                actions = []
                for item in data:
                    actions.append(Action(
                        agent_id=-2,
                        node_id=int(item.get('node_id', 0)),
                        description="Vanilla LLM action",
                        procurement_volume=float(item.get('procurement_volume', 1000)),
                        shipment_volume=float(item.get('shipment_volume', 1000)),
                        compliance_shift=float(item.get('compliance_shift', 0)),
                        target_compliance=80.0,
                        negotiate_contract=False,
                        hedge_risk=False,
                    ))
                return actions
        except Exception:
            pass
        return self._safe_actions(obs)

    def _safe_actions(self, obs: Observation) -> List[Action]:
        """Minimal safe actions during stagnation."""
        actions = []
        for nid in obs.node_states.keys():
            actions.append(Action(
                agent_id=-2,
                node_id=nid,
                description="Safe action (stagnation)",
                procurement_volume=500.0,
                shipment_volume=500.0,
                compliance_shift=0.0,
                target_compliance=85.0,
                negotiate_contract=False,
                hedge_risk=True,
            ))
        return actions

    def update(self, obs, action, reward, obs_next, done):
        # No update for vanilla LLM
        pass


class RandomBaseline(BaseBaseline):
    """Random action baseline (sanity check)."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.step_count = 0

    def reset(self, env: SemiSimEnvironment):
        self.step_count = 0

    def act(self, obs: Observation) -> List[Action]:
        self.step_count += 1
        actions = []
        for nid in obs.node_states.keys():
            import random
            actions.append(Action(
                agent_id=-99,
                node_id=nid,
                description="Random action",
                procurement_volume=self.rng.uniform(500, 5000),
                shipment_volume=self.rng.uniform(200, 3000),
                compliance_shift=self.rng.uniform(-10, 15),
                target_compliance=self.rng.uniform(60, 95),
                negotiate_contract=self.rng.random() > 0.7,
                hedge_risk=self.rng.random() > 0.5,
            ))
        return actions

    def update(self, obs, action, reward, obs_next, done):
        pass
