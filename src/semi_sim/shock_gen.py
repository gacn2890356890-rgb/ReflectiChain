"""
Shock Generator: Random / Adversarial / Historical shock injection.
"""
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from config import ShockConfig


class ShockGenerator:
    """
    Generates policy Black Swan events for Semi-Sim.
    Supports three modes:
        - Random: uniform random sampling from shock distribution
        - Adversarial: LLM-driven targeted policy generation
        - Historical: parameter-matched real-world events
    """

    def __init__(self, cfg: ShockConfig, seed: int = 42):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.attack_history: List[Dict] = []

    def generate_random_shock(self) -> Tuple[Optional[str], float]:
        """Generate a random shock event."""
        if self.rng.random() > self.cfg.shock_probability:
            return None, 0.0

        shock_type = self.rng.choice(self.cfg.shock_types)
        severity = max(0.3, min(1.0,
            self.cfg.shock_intensity_mean +
            self.rng.normal(0, self.cfg.shock_intensity_std)
        ))
        return shock_type, severity

    def generate_adversarial_shock(
        self,
        env_state: Dict[int, Any],
        llm_generator=None,
    ) -> Tuple[Optional[str], float, str]:
        """
        Generate an adversarial shock that targets the system's weakest nodes.

        Args:
            env_state: current state of each node {node_id: NodeState}
            llm_generator: optional LLM for sophisticated attack generation

        Returns:
            shock_type, severity, attack_rationale
        """
        # ── Automated weakness analysis ──────────────────────────────────────
        node_weaknesses = {}
        for nid, state in env_state.items():
            # Compute weakness score: lower compliance + high risk = vulnerable
            weakness = (1.0 - state.compliance / 100.0) * 0.5 + state.risk_exposure * 0.5
            node_weaknesses[nid] = weakness

        # Sort by weakness
        sorted_nodes = sorted(node_weaknesses.items(), key=lambda x: x[1], reverse=True)

        # Choose attack strategy based on strength level
        if self.cfg.adversarial_attack_strength < 0.4:
            # Weak: target the weakest node only
            target_nodes = [sorted_nodes[0][0]] if sorted_nodes else []
            shock_type = self.rng.choice(['material_shortage', 'logistics_disruption'])
            severity = 0.4
        elif self.cfg.adversarial_attack_strength < 0.7:
            # Medium: target top 2 weak nodes
            target_nodes = [n[0] for n in sorted_nodes[:2]]
            shock_type = self.rng.choice(['export_ban', 'tech_embargo'])
            severity = 0.65
        else:
            # Strong: coordinated attack on all weak nodes + systemic shock
            target_nodes = [n[0] for n in sorted_nodes[:3]]
            shock_type = 'export_ban'  # Most damaging
            severity = 0.9

        # ── LLM-enhanced attack generation ───────────────────────────────────
        if llm_generator is not None:
            attack_rationale = llm_generator.generate_adversarial_rationale(
                node_weaknesses, target_nodes, shock_type
            )
        else:
            attack_rationale = (
                f"Targeting nodes {target_nodes} due to "
                f"compliance risk ({node_weaknesses.get(target_nodes[0], 0):.2f})"
            )

        # Log attack
        self.attack_history.append({
            'target_nodes': target_nodes,
            'shock_type': shock_type,
            'severity': severity,
            'rationale': attack_rationale,
        })

        return shock_type, severity, attack_rationale

    def generate_historical_shock(
        self,
        event_name: str,
        current_step: int,
    ) -> Tuple[bool, Dict]:
        """
        Check if a historical event should trigger at the current step.

        Returns:
            (should_trigger, event_params)
        """
        event_params = self.cfg.historical_events.get(event_name, {})
        trigger_step = event_params.get('step', -1)

        if current_step == trigger_step:
            return True, event_params
        return False, {}

    def get_all_historical_events(self) -> List[str]:
        return list(self.cfg.historical_events.keys())

    def get_attack_statistics(self) -> Dict:
        """Return statistics of all adversarial attacks generated."""
        if not self.attack_history:
            return {'total_attacks': 0}

        severities = [a['severity'] for a in self.attack_history]
        shock_counts: Dict[str, int] = {}
        for a in self.attack_history:
            shock_counts[a['shock_type']] = shock_counts.get(a['shock_type'], 0) + 1

        return {
            'total_attacks': len(self.attack_history),
            'avg_severity': np.mean(severities),
            'max_severity': max(severities),
            'shock_distribution': shock_counts,
        }


class AdversarialPolicyLLM:
    """
    LLM-powered adversarial policy generator.
    Analyzes system state and generates targeted policy attacks.
    """

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def generate_adversarial_rationale(
        self,
        node_weaknesses: Dict[int, float],
        target_nodes: List[int],
        base_shock_type: str,
    ) -> str:
        """Generate an LLM-powered adversarial rationale via Ollama."""
        if self.llm is None:
            return f"Coordinated {base_shock_type} targeting nodes {target_nodes}"

        prompt = f"""
You are an adversarial policy analyst. Given the following supply chain vulnerabilities:

{chr(10).join([f"Node {nid}: weakness_score={w:.3f}" for nid, w in sorted(node_weaknesses.items(), key=lambda x: x[1], reverse=True)[:3]])}

Select the most damaging policy shock from: export_ban, material_shortage, financial_sanction, tech_embargo, demand_shock.

Generate a 1-sentence rationale for why this shock would be maximally effective against the identified vulnerabilities.
"""
        try:
            response = self.llm.generate(prompt, max_tokens=100, temperature=0.3)
            return response
        except Exception:
            return f"{base_shock_type} targeting weakest nodes: {target_nodes}"
