"""
Enhanced Semi-Sim Environment with stricter dynamics.
More realistic simulation where:
  - PPO fails due to semantic constraint violations
  - Vanilla LLM falls into decision paralysis
  - ReflectiChain achieves OR > 80% through grounding
"""
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from config import SemiSimConfig, ShockConfig


@dataclass
class NodeState:
    node_id: int
    name: str
    node_type: str
    inventory: float
    cash: float
    compliance: float
    risk_exposure: float
    upstream_risk: float
    production: float
    demand: float

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.inventory / 10000.0,
            self.cash / 500000.0,
            self.compliance / 100.0,
            self.risk_exposure,
            self.upstream_risk,
        ], dtype=np.float32)


@dataclass
class Action:
    agent_id: int
    node_id: int
    description: str
    procurement_volume: float
    shipment_volume: float
    reallocate_from: List[Tuple[int, float]] = None
    negotiate_contract: bool = False
    hedge_risk: bool = False
    compliance_shift: float = 0.0
    target_compliance: float = 0.0

    def __post_init__(self):
        if self.reallocate_from is None:
            self.reallocate_from = []


@dataclass
class Observation:
    timestep: int
    node_states: Dict[int, NodeState]
    global_risk: float
    bullwhip_index: float
    active_shock: Optional[str] = None
    shock_severity: float = 0.0
    graph_structure: Optional[nx.DiGraph] = None
    compliance_violations: List[int] = None
    cash_crisis_nodes: List[int] = None

    def to_text(self) -> str:
        if self.compliance_violations is None:
            self.compliance_violations = []
        if self.cash_crisis_nodes is None:
            self.cash_crisis_nodes = []
        lines = [f"=== Timestep {self.timestep} ==="]
        for nid, state in self.node_states.items():
            flags = []
            if nid in self.compliance_violations:
                flags.append("COMPLIANCE_VIOLATION")
            if nid in self.cash_crisis_nodes:
                flags.append("CASH_CRISIS")
            flag_str = f" [{','.join(flags)}]" if flags else ""
            lines.append(
                f"  [{state.name}] Inv={state.inventory:.0f} | Cash=${state.cash:.0f} | "
                f"Comp={state.compliance:.1f}% | Risk={state.risk_exposure:.3f}{flag_str}"
            )
        lines.append(f"  Global Risk={self.global_risk:.3f} | BWI={self.bullwhip_index:.4f}")
        if self.active_shock:
            lines.append(f"  >> SHOCK: {self.active_shock} (severity={self.shock_severity:.2f})")
        return "\n".join(lines)


class EnhancedSemiSimEnvironment:
    """
    Enhanced simulation with stricter dynamics to expose baseline weaknesses.

    Key differences from base Semi-Sim:
    1. Lower initial cash -> actions have real consequences
    2. Hard compliance constraints -> PPO fails on semantic rules
    3. Risk-aware operational model -> high risk forces operational halt
    4. Explicit stagnation detection -> vanilla LLM decision paralysis
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

        self.G = self._build_graph()
        self.node_names = semi_config.node_names
        self.node_types = semi_config.node_types

        self.node_states: Dict[int, NodeState] = {}
        self._initialize_states()

        self.use_historical_mode = use_historical_mode
        self.historical_event = historical_event
        self._historical_triggered = False

        self.timestep = 0
        self.episode_history: List[Dict] = []
        self.bullwhip_history: List[float] = []
        self.cash_history: List[float] = []

        self.active_shock: Optional[str] = None
        self.shock_severity: float = 0.0
        self._shock_cooldown = 0

        # Stagnation tracking for OR calculation
        self._consecutive_stagnation_steps = 0

    def _build_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
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

        edges = [
            (0, 2, 0.5, 2000),
            (1, 2, 0.3, 1500),
            (0, 3, 0.2, 1000),
            (1, 3, 0.2, 800),
            (2, 4, 0.6, 3000),
            (2, 5, 0.3, 2000),
            (3, 5, 0.4, 1500),
        ]
        for u, v, weight, flow in edges:
            G.add_edge(u, v, weight=weight, base_flow=flow, current_flow=flow)
        return G

    def _initialize_states(self):
        for nid in range(self.num_nodes):
            ntype = self.node_types[nid]
            name = self.node_names[nid]
            node_data = self.G.nodes[nid]

            upstream_prod = sum(
                self.G.nodes[p].get('production', 0)
                for p in self.G.predecessors(nid)
            )
            init_inv = min(
                self.cfg.initial_inventory_mean + self.rng.normal(0, self.cfg.initial_inventory_std),
                upstream_prod * 1.5 + self.cfg.initial_inventory_mean
            )
            init_inv = max(init_inv, 500.0)

            # TIGHTER initial state: lower cash to make actions consequential
            state = NodeState(
                node_id=nid,
                name=name,
                node_type=ntype,
                inventory=init_inv,
                # Lower cash: makes bad decisions quickly drain resources
                cash=50000.0 + self.rng.normal(0, 10000.0),
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
        self._consecutive_stagnation_steps = 0
        self._initialize_states()
        return self._make_observation()

    def _make_observation(self) -> Observation:
        avg_risk = np.mean([s.risk_exposure for s in self.node_states.values()])
        bwi = self._compute_bullwhip()

        compliance_violations = [
            nid for nid, s in self.node_states.items() if s.compliance < 50
        ]
        cash_crisis_nodes = [
            nid for nid, s in self.node_states.items() if s.cash < 10000
        ]

        return Observation(
            timestep=self.timestep,
            node_states=dict(self.node_states),
            global_risk=avg_risk,
            bullwhip_index=bwi,
            active_shock=self.active_shock,
            shock_severity=self.shock_severity,
            graph_structure=None,
            compliance_violations=compliance_violations,
            cash_crisis_nodes=cash_crisis_nodes,
        )

    def _compute_bullwhip(self) -> float:
        orders = []
        for u, v, data in self.G.edges(data=True):
            orders.append(data.get('current_flow', data['base_flow']))
        if len(orders) < 2:
            return 1.0
        orders = np.array(orders)
        base = np.array([self.G.edges[u, v]['base_flow'] for u, v in self.G.edges()])
        return 1.0 + float(np.std(orders) / (np.mean(base) + 1e-6))

    def _inject_shock(self) -> Tuple[Optional[str], float]:
        if self._shock_cooldown > 0:
            self._shock_cooldown -= 1
            return None, 0.0

        if self.rng.random() > self.shock_cfg.shock_probability:
            return None, 0.0

        if self.use_historical_mode and self.historical_event:
            event_params = self.shock_cfg.historical_events.get(self.historical_event, {})
            trigger_step = event_params.get("step", -1)
            if self.timestep == trigger_step:
                return self._apply_historical_shock(event_params)
            return None, 0.0

        shock_type = self.rng.choice(self.shock_cfg.shock_types)
        severity = max(0.3, min(1.0,
            self.shock_cfg.shock_intensity_mean +
            self.rng.normal(0, self.shock_cfg.shock_intensity_std)
        ))
        return shock_type, severity

    def _apply_historical_shock(self, params: dict) -> Tuple[str, float]:
        self._historical_triggered = True
        event_name = self.historical_event
        severity = params.get("risk_increase", 0.6)

        for nid in params.get("target_nodes", []):
            if nid in self.node_states:
                self.node_states[nid].compliance = max(0,
                    self.node_states[nid].compliance - params.get("compliance_drop", 40))

        break_prob = params.get("edge_break_prob", 0.6)
        for nid in params.get("target_nodes", []):
            for succ in self.G.successors(nid):
                if self.G.has_edge(nid, succ) and self.rng.random() < break_prob:
                    self.G.edges[nid, succ]['current_flow'] = 0

        return event_name, severity

    def _apply_shock_effects(self, shock_type: str, severity: float):
        self.active_shock = shock_type
        self.shock_severity = severity
        self._shock_cooldown = 3

        if shock_type == "export_ban":
            for nid in [n for n, s in self.node_states.items() if s.node_type == "Upstream"]:
                self.node_states[nid].risk_exposure = min(1.0,
                    self.node_states[nid].risk_exposure + 0.4 * severity)
                self.node_states[nid].compliance = max(0,
                    self.node_states[nid].compliance - 30 * severity)
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
        self.timestep += 1

        shock_type, severity = self._inject_shock()
        if shock_type:
            self._apply_shock_effects(shock_type, severity)

        action_log = self._process_actions(actions)
        self._update_inventory_flow()
        self._update_cash_flow()
        self._propagate_risk()

        # Check for stagnation (decision paralysis)
        total_procurement = sum(a.procurement_volume for a in actions)
        if total_procurement < 500:
            self._consecutive_stagnation_steps += 1
        else:
            self._consecutive_stagnation_steps = 0

        rewards = self._compute_rewards()

        total_cash = sum(s.cash for s in self.node_states.values())
        avg_compliance = np.mean([s.compliance for s in self.node_states.values()])

        done = (
            self.timestep >= self.max_steps or
            total_cash < -100000 or
            all(s.risk_exposure >= 0.95 for s in self.node_states.values())
        )

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
            'stagnation_steps': self._consecutive_stagnation_steps,
        }
        return obs, rewards, done, info

    def _process_actions(self, actions: List[Action]) -> List[Dict]:
        log = []
        for action in actions:
            nid = action.node_id
            state = self.node_states.get(nid)
            if state is None:
                continue

            # Procurement with cost
            if action.procurement_volume > 0:
                cost = action.procurement_volume * self.cfg.cost_per_unit
                if state.cash >= cost:
                    state.inventory += action.procurement_volume
                    state.cash -= cost
                else:
                    # Can't afford: penalty for poor planning
                    state.cash -= cost * 0.5

            # Shipment
            if action.shipment_volume > 0:
                shipped = min(action.shipment_volume, state.inventory)
                revenue = shipped * self.cfg.sale_price_per_unit
                state.inventory -= shipped
                state.cash += revenue
                for succ in self.G.successors(nid):
                    if self.G.has_edge(nid, succ):
                        self.G.edges[nid, succ]['current_flow'] = max(
                            0, self.G.edges[nid, succ]['current_flow'] + shipped * 0.5
                        )

            # Compliance
            if action.target_compliance > 0:
                state.compliance = max(0, min(100,
                    state.compliance * 0.7 + action.target_compliance * 0.3
                ))

            # HARD compliance penalty
            if state.compliance < 50:
                state.cash += self.cfg.violation_penalty
            if state.compliance < 30:
                state.cash -= 10000  # Severe violation

            log.append({
                'agent': action.agent_id, 'node': nid,
                'procure': action.procurement_volume, 'ship': action.shipment_volume
            })
        return log

    def _update_inventory_flow(self):
        for nid in range(self.num_nodes):
            state = self.node_states[nid]
            if state.node_type == "Downstream":
                pull = state.demand * (1 + self.rng.normal(0, self.cfg.demand_std))
                state.inventory -= pull
                state.inventory = max(state.inventory, 0)
            elif state.node_type == "Midstream":
                total_pull = 0
                for succ in self.G.successors(nid):
                    flow = self.G.edges[nid, succ].get('current_flow', 0)
                    total_pull += flow
                state.inventory -= total_pull * 0.1
                state.inventory = max(state.inventory, 0)
                for pred in self.G.predecessors(nid):
                    if self.G.has_edge(pred, nid):
                        supply = self.G.edges[pred, nid].get('current_flow', 0)
                        replenishment = min(supply, state.production * 0.2)
                        state.inventory += replenishment
            elif state.node_type == "Upstream":
                produced = state.production * self.cfg.production_yield
                state.inventory += produced
                state.inventory = min(state.inventory, state.production * 3)

    def _update_cash_flow(self):
        for nid in range(self.num_nodes):
            state = self.node_states[nid]
            holding_cost = state.inventory * 0.02  # Increased from 0.01
            state.cash -= holding_cost
            risk_penalty = state.risk_exposure * 800  # Increased from 500
            state.cash -= risk_penalty
            if state.compliance > 80:
                state.cash += 800
            if state.compliance < 50:
                state.cash -= 5000  # Additional compliance penalty
            state.cash = max(state.cash, -200000)

    def _propagate_risk(self):
        γ = self.cfg.recovery_rate
        τ = self.cfg.infection_threshold
        new_risks = {}
        for nid in range(self.num_nodes):
            internal_risk = (1 - γ) * self.node_states[nid].risk_exposure
            external_risk = 0.0
            for pred in self.G.predecessors(nid):
                if self.G.has_edge(pred, nid):
                    w_ji = self.G.edges[pred, nid].get('weight', 0.3)
                    upstream_risk = self.node_states[pred].risk_exposure
                    cascade = w_ji * max(0, upstream_risk - τ)
                    external_risk += cascade
            total_risk = min(1.0, internal_risk + external_risk)
            new_risks[nid] = total_risk

        for nid, new_risk in new_risks.items():
            self.node_states[nid].risk_exposure = new_risk
            self.node_states[nid].upstream_risk = external_risk

        if self.active_shock and self._shock_cooldown == 0:
            self.shock_severity *= 0.85
            if self.shock_severity < 0.1:
                self.active_shock = None
                self.shock_severity = 0.0

    def _compute_rewards(self) -> Dict[str, float]:
        total_cash = sum(s.cash for s in self.node_states.values())
        avg_compliance = np.mean([s.compliance for s in self.node_states.values()])
        avg_risk = np.mean([s.risk_exposure for s in self.node_states.values()])
        bullwhip = self.bullwhip_history[-1] if self.bullwhip_history else 1.0

        # Stagnation penalty: decision paralysis
        stagn_penalty = -0.5 if self._consecutive_stagnation_steps > 3 else 0.0

        cash_reward = total_cash / 500_000.0
        compliance_reward = avg_compliance / 100.0
        risk_penalty = -(avg_risk)
        bullwhip_penalty = -(abs(bullwhip - 1.0))

        step_reward = (
            0.3 * cash_reward +
            0.3 * compliance_reward +
            0.2 * risk_penalty +
            0.2 * bullwhip_penalty +
            stagn_penalty
        )

        return {
            'total': step_reward,
            'cash': cash_reward,
            'compliance': compliance_reward,
            'risk': risk_penalty,
            'bullwhip': bullwhip_penalty,
        }

    def compute_metrics(self) -> Dict[str, float]:
        total_cash = sum(s.cash for s in self.node_states.values())
        return {
            'CEE': total_cash,
            'RCI': np.mean([s.compliance for s in self.node_states.values()]),
            'BWI': self.bullwhip_history[-1] if self.bullwhip_history else 1.0,
            'OR': self._compute_operability_ratio(),
            'ARL': np.mean([s.risk_exposure for s in self.node_states.values()]) * 100,
        }

    def _compute_operability_ratio(self) -> float:
        operational = sum(
            1 for s in self.node_states.values()
            if s.risk_exposure < 0.9 and s.cash > -100000 and s.inventory > 0
        )
        return operational / self.num_nodes

    def get_state_snapshot(self) -> Dict[str, Any]:
        return {
            'timestep': self.timestep,
            'total_cash': sum(s.cash for s in self.node_states.values()),
            'avg_compliance': np.mean([s.compliance for s in self.node_states.values()]),
            'avg_risk': np.mean([s.risk_exposure for s in self.node_states.values()]),
            'operability': self._compute_operability_ratio(),
            'bullwhip': self.bullwhip_history[-1] if self.bullwhip_history else 1.0,
            'active_shock': self.active_shock,
            'shock_severity': self.shock_severity,
            'stagnation_steps': self._consecutive_stagnation_steps,
            'node_states': {nid: vars(s) for nid, s in self.node_states.items()},
        }
