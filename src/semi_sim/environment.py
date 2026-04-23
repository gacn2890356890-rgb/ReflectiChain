"""
Semi-Sim: High-fidelity Semiconductor Supply Chain Simulation Environment.
Implements POMDP-based risk propagation dynamics with SIR-inspired cascading model.
"""
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from config import SemiSimConfig, ShockConfig


@dataclass
class NodeState:
    """State of a single supply chain node."""
    node_id: int
    name: str
    node_type: str  # Upstream / Midstream / Downstream
    inventory: float      # Ii,t
    cash: float          # Ci,t
    compliance: float     # Ωi,t  [0, 100]
    risk_exposure: float  # Ri,t  [0, 1]
    upstream_risk: float  # Accumulated upstream risk
    production: float     # Production capacity
    demand: float         # Downstream demand

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.inventory / 10000.0,
            self.cash / 500000.0,
            self.compliance / 100.0,
            self.risk_exposure,
            self.upstream_risk,
        ], dtype=np.float32)

    def __repr__(self):
        return (f"Node({self.name}, I={self.inventory:.0f}, C={self.cash:.0f}, "
                f"Ω={self.compliance:.1f}, R={self.risk_exposure:.3f})")


@dataclass
class Action:
    """An agent's supply chain intervention at a given timestep."""
    agent_id: int
    node_id: int
    description: str
    procurement_volume: float   # qbuy
    shipment_volume: float      # qship
    reallocate_from: List[Tuple[int, float]] = field(default_factory=list)  # (node_id, volume)
    negotiate_contract: bool = False
    hedge_risk: bool = False
    compliance_shift: float = 0.0  # Δcompliance
    target_compliance: float = 0.0  # Target compliance after action


@dataclass
class Observation:
    """POMDP observation at a timestep."""
    timestep: int
    node_states: Dict[int, NodeState]
    global_risk: float        # System-wide average risk
    bullwhip_index: float     # BWI
    active_shock: Optional[str] = None
    shock_severity: float = 0.0
    graph_structure: Optional[nx.DiGraph] = None

    def to_text(self) -> str:
        lines = [f"=== Timestep {self.timestep} ==="]
        for nid, state in self.node_states.items():
            lines.append(
                f"  [{state.name}] Inv={state.inventory:.0f} | Cash=${state.cash:.0f} | "
                f"Comp={state.compliance:.1f}% | Risk={state.risk_exposure:.3f}"
            )
        lines.append(f"  Global Risk={self.global_risk:.3f} | BWI={self.bullwhip_index:.4f}")
        if self.active_shock:
            lines.append(f"  >> SHOCK: {self.active_shock} (severity={self.shock_severity:.2f})")
        return "\n".join(lines)


class SemiSimEnvironment:
    """
    High-fidelity semiconductor supply chain simulation environment.

    State: Graph G=(V,E) where each node has (inventory, cash, compliance, risk_exposure).
    Dynamics: SIR-inspired risk propagation with policy-driven volatility.
    """

    def __init__(
        self,
        semi_config: SemiSimConfig,
        shock_config: ShockConfig,
        seed: int = 42,
        use_historical_mode: bool = False,
        historical_event: Optional[str] = None,
    ):
        self.cfg = semi_config
        self.shock_cfg = shock_config
        self.rng = np.random.default_rng(seed)

        self.num_nodes = semi_config.num_nodes
        self.max_steps = semi_config.max_steps

        # Build supply chain graph
        self.G = self._build_graph()
        self.node_names = semi_config.node_names
        self.node_types = semi_config.node_types

        # Node states indexed by node_id
        self.node_states: Dict[int, NodeState] = {}
        self._initialize_states()

        # Historical mode
        self.use_historical_mode = use_historical_mode
        self.historical_event = historical_event
        self._historical_triggered = False

        # Tracking metrics
        self.timestep = 0
        self.episode_history: List[Dict] = []
        self.bullwhip_history: List[float] = []
        self.cash_history: List[float] = []

        # Shock state
        self.active_shock: Optional[str] = None
        self.shock_severity: float = 0.0

        # Cooldown to prevent shock stacking
        self._shock_cooldown = 0

    def _build_graph(self) -> nx.DiGraph:
        """Build the supply chain network topology."""
        G = nx.DiGraph()

        # Node definitions: (id, type, production_capacity, base_demand)
        nodes = [
            (0, "Upstream", 10000, 0),
            (1, "Upstream", 8000, 0),
            (2, "Midstream", 6000, 0),
            (3, "Midstream", 5000, 0),
            (4, "Downstream", 0, 4000),
            (5, "Downstream", 0, 3000),
        ]

        for nid, ntype, prod, demand in nodes:
            G.add_node(nid, type=ntype, production=prod, base_demand=demand)

        # Edge definitions: (from, to, weight, base_flow)
        edges = [
            (0, 2, 0.5, 2000),   # ASML -> TSMC
            (1, 2, 0.3, 1500),   # Shin-Etsu -> TSMC
            (0, 3, 0.2, 1000),   # ASML -> Samsung
            (1, 3, 0.2, 800),
            (2, 4, 0.6, 3000),   # TSMC -> Apple
            (2, 5, 0.3, 2000),   # TSMC -> NVIDIA
            (3, 5, 0.4, 1500),   # Samsung -> NVIDIA
        ]

        for u, v, weight, flow in edges:
            G.add_edge(u, v, weight=weight, base_flow=flow, current_flow=flow)

        return G

    def _initialize_states(self):
        """Initialize node states at episode start."""
        for nid in range(self.num_nodes):
            ntype = self.node_types[nid]
            name = self.node_names[nid]
            node_data = self.G.nodes[nid]

            # Compute initial inventory from upstream production capacity
            upstream_prod = sum(
                self.G.nodes[p].get('production', 0)
                for p in self.G.predecessors(nid)
            )
            init_inv = min(
                self.cfg.initial_inventory_mean + self.rng.normal(0, self.cfg.initial_inventory_std),
                upstream_prod * 1.5 + self.cfg.initial_inventory_mean
            )
            init_inv = max(init_inv, 500.0)

            state = NodeState(
                node_id=nid,
                name=name,
                node_type=ntype,
                inventory=init_inv,
                cash=self.cfg.initial_cash_mean + self.rng.normal(0, self.cfg.initial_cash_std),
                compliance=self.cfg.initial_compliance_mean + self.rng.uniform(-5, 5),
                risk_exposure=max(0.0, min(1.0,
                    self.cfg.initial_risk_mean + self.cfg.initial_risk_offset +
                    self.rng.normal(0, 0.05)
                )),
                upstream_risk=0.0,
                production=node_data.get('production', 0),
                demand=node_data.get('base_demand', 0),
            )
            state.compliance = max(0.0, min(100.0, state.compliance))
            self.node_states[nid] = state

    def reset(self, seed: Optional[int] = None) -> Observation:
        """Reset the environment to initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.timestep = 0
        self.episode_history = []
        self.bullwhip_history = []
        self.cash_history = []
        self.active_shock = None
        self.shock_severity = 0.0
        self._shock_cooldown = 0
        self._historical_triggered = False

        self._initialize_states()
        return self._make_observation()

    def _make_observation(self) -> Observation:
        """Construct POMDP observation from current state."""
        # Compute global risk
        avg_risk = np.mean([s.risk_exposure for s in self.node_states.values()])

        # Compute bullwhip index (variance of order fluctuations)
        bwi = self._compute_bullwhip()

        return Observation(
            timestep=self.timestep,
            node_states=dict(self.node_states),
            global_risk=avg_risk,
            bullwhip_index=bwi,
            active_shock=self.active_shock,
            shock_severity=self.shock_severity,
            graph_structure=None,
        )

    def _compute_bullwhip(self) -> float:
        """Compute Bullwhip Effect Index: variance of order quantities."""
        orders = []
        for eid, (u, v, data) in enumerate(self.G.edges(data=True)):
            orders.append(data.get('current_flow', data['base_flow']))
        if len(orders) < 2:
            return 1.0
        orders = np.array(orders)
        base = np.array([self.G.edges[u, v]['base_flow'] for u, v in self.G.edges()])
        # BWI = 1.0 means perfect stability; >1 means amplified oscillations
        return 1.0 + float(np.std(orders) / (np.mean(base) + 1e-6))

    def _inject_shock(self) -> Tuple[Optional[str], float]:
        """Inject a random policy black swan shock."""
        if self._shock_cooldown > 0:
            self._shock_cooldown -= 1
            return None, 0.0

        if self.rng.random() > self.shock_cfg.shock_probability:
            return None, 0.0

        # Historical mode: trigger specific event
        if self.use_historical_mode and self.historical_event:
            event_params = self.shock_cfg.historical_events.get(self.historical_event, {})
            trigger_step = event_params.get("step", -1)
            if self.timestep == trigger_step:
                return self._apply_historical_shock(event_params)
            return None, 0.0

        # Random shock from available types
        shock_type = self.rng.choice(self.shock_cfg.shock_types)
        severity = max(0.3, min(1.0,
            self.shock_cfg.shock_intensity_mean +
            self.rng.normal(0, self.shock_cfg.shock_intensity_std)
        ))

        return shock_type, severity

    def _apply_historical_shock(self, params: dict) -> Tuple[str, float]:
        """Apply a historical event shock."""
        self._historical_triggered = True
        event_name = self.historical_event
        severity = params.get("risk_increase", 0.6)

        # Apply compliance drops
        for nid in params.get("target_nodes", []):
            if nid in self.node_states:
                self.node_states[nid].compliance = max(0,
                    self.node_states[nid].compliance - params.get("compliance_drop", 40))

        # Apply edge break probability
        break_prob = params.get("edge_break_prob", 0.6)
        edges_to_break = []
        for nid in params.get("target_nodes", []):
            for succ in self.G.successors(nid):
                if self.rng.random() < break_prob:
                    edges_to_break.append((nid, succ))
        for u, v in edges_to_break:
            self.G.edges[u, v]['current_flow'] = 0

        return event_name, severity

    def _apply_shock_effects(self, shock_type: str, severity: float):
        """Apply the effects of an active shock to the environment."""
        self.active_shock = shock_type
        self.shock_severity = severity
        self._shock_cooldown = 3  # Prevent immediate re-trigger

        if shock_type == "export_ban":
            # Target upstream nodes
            for nid in [n for n, s in self.node_states.items() if s.node_type == "Upstream"]:
                self.node_states[nid].risk_exposure = min(1.0,
                    self.node_states[nid].risk_exposure + 0.4 * severity)
                self.node_states[nid].compliance = max(0,
                    self.node_states[nid].compliance - 30 * severity)
                # Disrupt downstream flow
                for succ in self.G.successors(nid):
                    if self.G.has_edge(nid, succ):
                        self.G.edges[nid, succ]['current_flow'] *= (1 - 0.7 * severity)

        elif shock_type == "material_shortage":
            for nid in [n for n, s in self.node_states.items() if s.node_type == "Midstream"]:
                self.node_states[nid].inventory *= (1 - 0.5 * severity)
                self.node_states[nid].risk_exposure = min(1.0,
                    self.node_states[nid].risk_exposure + 0.3 * severity)

        elif shock_type == "financial_sanction":
            for nid in self.node_states:
                self.node_states[nid].cash *= (1 - 0.4 * severity)
                self.node_states[nid].risk_exposure = min(1.0,
                    self.node_states[nid].risk_exposure + 0.2 * severity)

        elif shock_type == "tech_embargo":
            for nid in [n for n, s in self.node_states.items() if s.node_type == "Midstream"]:
                self.node_states[nid].compliance = max(0,
                    self.node_states[nid].compliance - 50 * severity)
                self.node_states[nid].production *= (1 - 0.6 * severity)

        elif shock_type == "demand_shock":
            for nid in [n for n, s in self.node_states.items() if s.node_type == "Downstream"]:
                self.node_states[nid].demand *= (1 - 0.5 * severity)
                # Propagate upstream
                for pred in self.G.predecessors(nid):
                    if self.G.has_edge(pred, nid):
                        self.G.edges[pred, nid]['current_flow'] *= (1 - 0.4 * severity)

        elif shock_type == "logistics_disruption":
            for u, v, data in self.G.edges(data=True):
                if self.rng.random() < 0.3 * severity:
                    data['current_flow'] *= (1 - 0.4 * severity)
            for nid in self.node_states:
                self.node_states[nid].risk_exposure = min(1.0,
                    self.node_states[nid].risk_exposure + 0.15 * severity)

    def step(self, actions: List[Action]) -> Tuple[Observation, Dict[str, float], bool, Dict]:
        """
        Execute one step of the simulation.

        Args:
            actions: List of actions from agent(s)

        Returns:
            obs: POMDP observation
            rewards: Dict of reward signals
            done: Episode termination flag
            info: Additional debug info
        """
        self.timestep += 1

        # 1. Inject shock
        shock_type, severity = self._inject_shock()
        if shock_type:
            self._apply_shock_effects(shock_type, severity)

        # 2. Process actions
        action_log = self._process_actions(actions)

        # 3. Update supply chain dynamics
        self._update_inventory_flow()
        self._update_cash_flow()

        # 4. Risk propagation (SIR-inspired)
        self._propagate_risk()

        # 5. Compute rewards
        rewards = self._compute_rewards()

        # 6. Check termination
        total_cash = sum(s.cash for s in self.node_states.values())
        avg_compliance = np.mean([s.compliance for s in self.node_states.values()])
        done = (
            self.timestep >= self.max_steps or
            total_cash < -100000 or           # Systemic bankruptcy
            all(s.risk_exposure >= 0.95 for s in self.node_states.values())  # Total collapse
        )

        # 7. Track history
        self.bullwhip_history.append(self._compute_bullwhip())
        self.cash_history.append(total_cash)
        self.episode_history.append({
            'timestep': self.timestep,
            'shock': shock_type,
            'severity': severity,
            'rewards': rewards,
            'action_log': action_log,
            'node_states': {nid: s.to_vector() for nid, s in self.node_states.items()},
        })

        obs = self._make_observation()
        info = {
            'total_cash': total_cash,
            'avg_compliance': avg_compliance,
            'avg_risk': np.mean([s.risk_exposure for s in self.node_states.values()]),
            'bullwhip': self.bullwhip_history[-1],
        }
        return obs, rewards, done, info

    def _process_actions(self, actions: List[Action]) -> List[Dict]:
        """Execute supply chain actions."""
        log = []
        for action in actions:
            nid = action.node_id
            state = self.node_states.get(nid)
            if state is None:
                continue

            # Procurement
            if action.procurement_volume > 0:
                cost = action.procurement_volume * self.cfg.cost_per_unit
                if state.cash >= cost:
                    state.inventory += action.procurement_volume
                    state.cash -= cost

            # Shipment
            if action.shipment_volume > 0:
                shipped = min(action.shipment_volume, state.inventory)
                revenue = shipped * self.cfg.sale_price_per_unit
                state.inventory -= shipped
                state.cash += revenue
                # Update downstream flow
                for succ in self.G.successors(nid):
                    if self.G.has_edge(nid, succ):
                        self.G.edges[nid, succ]['current_flow'] = max(
                            0, self.G.edges[nid, succ]['current_flow'] + shipped * 0.5
                        )

            # Compliance adjustment
            if action.target_compliance > 0:
                state.compliance = max(0, min(100,
                    state.compliance * 0.7 + action.target_compliance * 0.3
                ))

            # Compliance violation penalty
            if state.compliance < 50:
                state.cash += self.cfg.violation_penalty

            log.append({
                'agent': action.agent_id, 'node': nid,
                'procure': action.procurement_volume,
                'ship': action.shipment_volume
            })
        return log

    def _update_inventory_flow(self):
        """Update inventory based on upstream supply and downstream demand."""
        # Compute demand signals propagating upstream
        for nid in range(self.num_nodes):
            state = self.node_states[nid]
            if state.node_type == "Downstream":
                # Pull model: downstream demand drives orders
                pull = state.demand * (1 + self.rng.normal(0, self.cfg.demand_std))
                state.inventory -= pull
                state.inventory = max(state.inventory, 0)
            elif state.node_type == "Midstream":
                # Pull from upstream
                total_pull = 0
                for succ in self.G.successors(nid):
                    flow = self.G.edges[nid, succ].get('current_flow', 0)
                    total_pull += flow
                state.inventory -= total_pull * 0.1
                state.inventory = max(state.inventory, 0)
                # Supply replenishment from upstream
                for pred in self.G.predecessors(nid):
                    if self.G.has_edge(pred, nid):
                        supply = self.G.edges[pred, nid].get('current_flow', 0)
                        replenishment = min(supply, state.production * 0.2)
                        state.inventory += replenishment
            elif state.node_type == "Upstream":
                # Natural production with yield loss
                produced = state.production * self.cfg.production_yield
                state.inventory += produced
                state.inventory = min(state.inventory, state.production * 3)  # Cap

    def _update_cash_flow(self):
        """Update cash based on inventory and demand."""
        for nid in range(self.num_nodes):
            state = self.node_states[nid]
            # Holding cost
            holding_cost = state.inventory * 0.01
            state.cash -= holding_cost
            # Risk penalty
            risk_penalty = state.risk_exposure * 500
            state.cash -= risk_penalty
            # Compliance reward
            if state.compliance > 80:
                state.cash += 1000
            # Cash cannot go below zero
            state.cash = max(state.cash, -200000)

    def _propagate_risk(self):
        """
        SIR-inspired risk propagation on the supply chain graph.
        Implements the dynamics from Equation (8) in the paper:
        Ri,t+1 = (1 - γ) * Ri,t + Σ_{j∈Npred(i)} wji * max(0, Rj,t - τ)
        """
        γ = self.cfg.recovery_rate
        τ = self.cfg.infection_threshold

        new_risks = {}
        for nid in range(self.num_nodes):
            # Endogenous attenuation: natural recovery
            internal_risk = (1 - γ) * self.node_states[nid].risk_exposure

            # Exogenous filtering: upstream cascade
            external_risk = 0.0
            for pred in self.G.predecessors(nid):
                if self.G.has_edge(pred, nid):
                    w_ji = self.G.edges[pred, nid].get('weight', 0.3)
                    upstream_risk = self.node_states[pred].risk_exposure
                    cascade = w_ji * max(0, upstream_risk - τ)
                    external_risk += cascade

            # Total risk
            total_risk = min(1.0, internal_risk + external_risk)
            new_risks[nid] = total_risk

        # Apply new risks
        for nid, new_risk in new_risks.items():
            self.node_states[nid].risk_exposure = new_risk
            self.node_states[nid].upstream_risk = external_risk

        # Decay shock
        if self.active_shock and self._shock_cooldown == 0:
            self.shock_severity *= 0.85
            if self.shock_severity < 0.1:
                self.active_shock = None
                self.shock_severity = 0.0

    def _compute_rewards(self) -> Dict[str, float]:
        """Compute multi-dimensional reward signals."""
        total_cash = sum(s.cash for s in self.node_states.values())
        avg_compliance = np.mean([s.compliance for s in self.node_states.values()])
        avg_risk = np.mean([s.risk_exposure for s in self.node_states.values()])

        # Normalized reward components
        cash_reward = total_cash / 1_000_000.0
        compliance_reward = avg_compliance / 100.0
        risk_penalty = -(avg_risk)  # Negative risk is reward
        bullwhip = self.bullwhip_history[-1] if self.bullwhip_history else 1.0
        bullwhip_penalty = -(abs(bullwhip - 1.0))

        # Combined step reward
        step_reward = (
            0.3 * cash_reward +
            0.3 * compliance_reward +
            0.2 * risk_penalty +
            0.2 * bullwhip_penalty
        )

        return {
            'total': step_reward,
            'cash': cash_reward,
            'compliance': compliance_reward,
            'risk': risk_penalty,
            'bullwhip': bullwhip_penalty,
        }

    # ─── Evaluation Metrics (for Table 1) ───────────────────────────────────
    def compute_metrics(self) -> Dict[str, float]:
        """Compute all metrics matching Table 1 in the paper."""
        total_cash = sum(s.cash for s in self.node_states.values())

        return {
            'CEE': total_cash,                      # Total Cash
            'RCI': np.mean([s.compliance for s in self.node_states.values()]),  # Avg Compliance
            'BWI': self.bullwhip_history[-1] if self.bullwhip_history else 1.0,  # Bullwhip Index
            'OR': self._compute_operability_ratio(),  # Operability Ratio
            'ARL': np.mean([s.risk_exposure for s in self.node_states.values()]) * 100,  # Avg Risk Level
        }

    def _compute_operability_ratio(self) -> float:
        """OR = fraction of nodes that are operational (not collapsed)."""
        operational = sum(
            1 for s in self.node_states.values()
            if s.risk_exposure < 0.9 and s.cash > -100000 and s.inventory > 0
        )
        return operational / self.num_nodes

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get a complete snapshot of the current state."""
        return {
            'timestep': self.timestep,
            'total_cash': sum(s.cash for s in self.node_states.values()),
            'avg_compliance': np.mean([s.compliance for s in self.node_states.values()]),
            'avg_risk': np.mean([s.risk_exposure for s in self.node_states.values()]),
            'operability': self._compute_operability_ratio(),
            'bullwhip': self.bullwhip_history[-1] if self.bullwhip_history else 1.0,
            'active_shock': self.active_shock,
            'shock_severity': self.shock_severity,
            'node_states': {nid: vars(s) for nid, s in self.node_states.items()},
        }
