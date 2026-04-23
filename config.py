"""
Global configuration for ReflectiChain project.
All hyperparameters are centralized here for easy tuning.
"""
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


# ─── LLM Configuration ────────────────────────────────────────────────────────
@dataclass
class LLMConfig:
    model_name: str = "qwen:7b"          # Ollama model name
    api_base: str = "http://localhost:11434"  # Ollama server
    api_key: str = "EMPTY"
    temperature_action: float = 0.8     # Action sampling temperature
    temperature_reflect: float = 0.3   # Reflection temperature
    max_tokens_action: int = 512
    max_tokens_reflect: int = 256
    top_p: float = 0.9
    use_ollama: bool = True             # Set False to use mock (no real LLM)


# ─── World Model Configuration ─────────────────────────────────────────────────
@dataclass
class WorldModelConfig:
    latent_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    trajectory_length: int = 5          # Latent rollout horizon K
    num_rollouts: int = 3              # Number of candidate rollouts
    device: str = "cuda"


# ─── Simulation Environment Configuration ─────────────────────────────────────
@dataclass
class SemiSimConfig:
    # Network topology
    num_nodes: int = 6                  # |V| = 6 critical entities
    num_edges: int = 7                  # |E| = 7 strategic links
    node_types: List[str] = field(default_factory=lambda: [
        "Upstream", "Upstream", "Midstream", "Midstream", "Downstream", "Downstream"
    ])
    node_names: List[str] = field(default_factory=lambda: [
        "ASML", "Shin-Etsu", "TSMC", "Samsung", "Apple", "NVIDIA"
    ])

    # Initial state
    initial_cash_mean: float = 150000.0
    initial_cash_std: float = 30000.0
    initial_inventory_mean: float = 5000.0
    initial_inventory_std: float = 1000.0
    initial_compliance_mean: float = 90.0
    initial_risk_mean: float = 0.2

    # Simulation dynamics
    max_steps: int = 30                 # T = 30 strategic gaming steps
    recovery_rate: float = 0.15          # γ - resilience recovery rate
    infection_threshold: float = 0.3     # τ - risk cascade trigger threshold
    cost_per_unit: float = 50.0         # pcost
    sale_price_per_unit: float = 120.0  # psale
    violation_penalty: float = -50000.0  # Γ - penalty for non-compliant actions

    # Initial instability boundary (deliberately weak)
    initial_risk_offset: float = -0.21

    # Supply chain transition parameters
    demand_std: float = 0.1
    production_yield: float = 0.95


# ─── Agent Configuration ───────────────────────────────────────────────────────
@dataclass
class AgentConfig:
    num_candidates: int = 3             # N - number of candidate actions
    memory_window: int = 3               # K - hindsight window size
    alpha_semantic: float = 0.3         # Weight for semantic score
    beta_physical: float = 0.7          # Weight for world model reward
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    learning_rate: float = 1e-4
    kl_divergence_weight: float = 0.01  # KL divergence regularization


# ─── Multi-Agent Configuration ─────────────────────────────────────────────────
@dataclass
class MultiAgentConfig:
    num_agents: int = 3
    agent_types: List[str] = field(default_factory=lambda: [
        "profit_driven", "resilience_driven", "compliance_driven"
    ])
    agent_names: List[str] = field(default_factory=lambda: [
        "Agent-A (Profit)", "Agent-B (Resilience)", "Agent-C (Compliance)"
    ])
    competition_intensity: float = 0.5  # Resource scarcity factor
    enable_cooperation: bool = True    # Whether cooperation protocols are allowed
    max_cooperation_rounds: int = 5


# ─── Shock / Scenario Configuration ────────────────────────────────────────────
@dataclass
class ShockConfig:
    shock_types: List[str] = field(default_factory=lambda: [
        "export_ban", "material_shortage", "financial_sanction",
        "tech_embargo", "demand_shock", "logistics_disruption"
    ])
    shock_probability: float = 0.15     # Probability of shock at each step
    shock_intensity_mean: float = 0.7    # Mean shock strength
    shock_intensity_std: float = 0.2

    # Adversarial settings
    adversarial_attack_strength: float = 0.8  # 0=weak, 1=strong
    adversarial_smart: bool = True

    # Historical event parameters (mapped to Semi-Sim)
    historical_events: dict = field(default_factory=lambda: {
        "entity_list_2019": {
            "step": 5, "target_nodes": [2], "compliance_drop": 40,
            "edge_break_prob": 0.6, "risk_increase": 0.5
        },
        "chips_act_2022": {
            "step": 10, "subsidy_amount": 80000,
            "cash_boost": [0, 1, 2, 3], "compliance_bonus": 10
        },
        "export_restriction_2022": {
            "step": 15, "target_nodes": [0, 1], "compliance_drop": 60,
            "risk_increase": 0.7, "edge_break_prob": 0.8
        },
        "gallium_germanium_2023": {
            "step": 20, "target_nodes": [1], "supply_weight_drop": 0.3,
            "inventory_impact": -2000, "risk_increase": 0.4
        }
    })


# ─── Evaluation Metrics Configuration ─────────────────────────────────────────
@dataclass
class EvalConfig:
    # Target values for Table 1 in the paper
    ideal_cash: float = 1_000_000.0
    ideal_compliance: float = 85.0
    ideal_operability: float = 80.0
    ideal_risk: float = 40.0
    bullwhip_target: Tuple[float, float] = (1.0, 1.5)

    # Game-theoretic thresholds
    nash_epsilon_tolerance: float = 0.1  # ε-NE threshold for "near-equilibrium"

    # Adversarial thresholds
    worst_case_percentile: float = 5.0   # WCR: survival in worst 5% scenarios
    recovery_operability_threshold: float = 80.0
