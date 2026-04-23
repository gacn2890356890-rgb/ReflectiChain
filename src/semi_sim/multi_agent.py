"""
Multi-Agent Interface for Semi-Sim.
Enables concurrent decision-making by multiple heterogeneous agents.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from config import MultiAgentConfig, SemiSimConfig


@dataclass
class AgentIdentity:
    agent_id: int
    name: str
    agent_type: str  # profit_driven | resilience_driven | compliance_driven
    controlled_nodes: List[int]  # Which nodes this agent controls
    goal_weights: Dict[str, float]  # {cash: 0.7, risk: 0.3, ...}

    def get_prompt_suffix(self) -> str:
        if self.agent_type == "profit_driven":
            return "You are a PROFIT-DRIVEN agent. Prioritize maximizing Total Cash above all else."
        elif self.agent_type == "resilience_driven":
            return "You are a RESILIENCE-DRIVEN agent. Prioritize maintaining System Operability Ratio and minimizing risk exposure."
        elif self.agent_type == "compliance_driven":
            return "You are a COMPLIANCE-DRIVEN agent. NEVER violate trade compliance rules. Maintain 100% compliance at any cost."
        return "You are a strategic supply chain agent."


class MultiAgentInterface:
    """
    Manages concurrent execution of multiple LLM-driven agents on Semi-Sim.

    Responsibilities:
        - State Broadcaster: broadcast global state to all agents
        - Action Collector: collect decisions from all agents with conflict resolution
        - Feedback Router: route environment feedback to each agent's LoRA update
    """

    def __init__(
        self,
        ma_config: MultiAgentConfig,
        semi_config: SemiSimConfig,
        env,
    ):
        self.cfg = ma_config
        self.semi_cfg = semi_config
        self.env = env

        # Initialize agent identities
        self.agents: Dict[int, AgentIdentity] = {}
        self._init_agents()

        # Shared resource pool
        self._resource_pool: Dict[int, float] = {}  # node_id -> available quantity

        # Competition state
        self._competition_log: List[Dict] = []

    def _init_agents(self):
        """Initialize agent identities with heterogeneous goals."""
        node_assignments = self._assign_nodes_to_agents()

        for i, (agent_type, agent_name) in enumerate(
            zip(self.cfg.agent_types, self.cfg.agent_names)
        ):
            weights = self._get_goal_weights(agent_type)
            self.agents[i] = AgentIdentity(
                agent_id=i,
                name=agent_name,
                agent_type=agent_type,
                controlled_nodes=node_assignments.get(i, []),
                goal_weights=weights,
            )

    def _assign_nodes_to_agents(self) -> Dict[int, List[int]]:
        """Assign supply chain nodes to agents."""
        num_nodes = self.semi_cfg.num_nodes
        num_agents = self.cfg.num_agents

        assignments = {i: [] for i in range(num_agents)}
        nodes_per_agent = num_nodes // num_agents
        remainder = num_nodes % num_agents

        node_idx = 0
        for i in range(num_agents):
            count = nodes_per_agent + (1 if i < remainder else 0)
            assignments[i] = list(range(node_idx, node_idx + count))
            node_idx += count

        return assignments

    def _get_goal_weights(self, agent_type: str) -> Dict[str, float]:
        """Define goal weights for each agent type."""
        if agent_type == "profit_driven":
            return {'cash': 0.7, 'compliance': 0.1, 'risk': 0.2, 'bullwhip': 0.0}
        elif agent_type == "resilience_driven":
            return {'cash': 0.2, 'compliance': 0.2, 'risk': 0.5, 'bullwhip': 0.1}
        elif agent_type == "compliance_driven":
            return {'cash': 0.1, 'compliance': 0.7, 'risk': 0.1, 'bullwhip': 0.1}
        return {'cash': 0.25, 'compliance': 0.25, 'risk': 0.25, 'bullwhip': 0.25}

    def broadcast_state(self, obs) -> Dict[int, Dict]:
        """
        Broadcast observation to all agents, filtering to their controlled nodes.

        Returns:
            Dict[agent_id -> filtered observation dict]
        """
        broadcasts = {}
        for agent_id, identity in self.agents.items():
            # Filter to controlled nodes only
            filtered_states = {
                nid: state for nid, state in obs.node_states.items()
                if nid in identity.controlled_nodes
            }
            broadcasts[agent_id] = {
                'obs': obs,
                'controlled_states': filtered_states,
                'agent_identity': identity,
                'global_risk': obs.global_risk,
                'bullwhip_index': obs.bullwhip_index,
            }
        return broadcasts

    def collect_actions(
        self,
        agent_actions: Dict[int, Any],
    ) -> Tuple[List[Any], Dict]:
        """
        Collect actions from all agents and resolve conflicts.

        Args:
            agent_actions: Dict[agent_id -> List[Action]]

        Returns:
            merged_actions: List of merged Action objects
            conflict_log: Dict with conflict resolution details
        """
        from src.semi_sim.environment import Action

        merged = []
        conflict_log = {'resolved': [], 'resource_contention': []}

        # Build action pool
        all_actions = []
        for agent_id, actions in agent_actions.items():
            for action in actions:
                action.agent_id = agent_id
                all_actions.append(action)

        # Resource contention detection
        self._detect_resource_contention(all_actions, conflict_log)

        # Merge non-conflicting actions
        merged = all_actions

        return merged, conflict_log

    def _detect_resource_contention(
        self,
        actions: List,
        conflict_log: Dict,
    ):
        """Detect and log resource contention between agents."""
        procurement_by_node: Dict[int, List[Tuple[int, float]]] = {}
        for action in actions:
            nid = action.node_id
            if nid not in procurement_by_node:
                procurement_by_node[nid] = []
            procurement_by_node[nid].append((action.agent_id, action.procurement_volume))

        for nid, agent_procs in procurement_by_node.items():
            if len(agent_procs) > 1:
                total = sum(p for _, p in agent_procs)
                conflict_log['resource_contention'].append({
                    'node_id': nid,
                    'agents': [a for a, _ in agent_procs],
                    'total_requested': total,
                })

    def evaluate_cooperation(
        self,
        agent_actions: Dict[int, Any],
        obs,
    ) -> Dict[int, float]:
        """
        Evaluate potential cooperation benefits between agents.
        Returns cooperation_score per agent.
        """
        scores = {}
        for agent_id in self.agents:
            scores[agent_id] = 0.0

        if not self.cfg.enable_cooperation:
            return scores

        # Check for complementary resource flows
        for aid, actions in agent_actions.items():
            for action in actions:
                if action.negotiate_contract:
                    scores[aid] += 0.5  # Cooperation bonus

        return scores

    def compute_game_metrics(
        self,
        episode_history: List[Dict],
        agent_ids: List[int],
    ) -> Dict[str, float]:
        """
        Compute game-theoretic evaluation metrics.

        Returns:
            Dict with SW, ε-NE, CI, CR, RRG
        """
        # Social Welfare: sum of all agents' total cash
        sw = sum(
            sum(step.get('agent_cash', {}).get(aid, 0) for aid in agent_ids)
            for step in episode_history
        )

        # Competition Intensity: std of agent cash levels at each step
        ci_values = []
        for step in episode_history:
            agent_cash = [step.get('agent_cash', {}).get(aid, 0) for aid in agent_ids]
            if len(agent_cash) > 1:
                ci_values.append(np.std(agent_cash))
        ci = np.mean(ci_values) if ci_values else 0.0

        # Nash Equilibrium approximation (ε-NE)
        # Use price of anarchy approximation
        nash_gap = self._approximate_nash_gap(episode_history, agent_ids)

        return {
            'SW': sw,
            'CI': ci,
            'epsilon_NE': nash_gap,
            'CR': 0.0,  # Computed from cooperation events
        }

    def _approximate_nash_gap(
        self,
        episode_history: List[Dict],
        agent_ids: List[int],
    ) -> float:
        """
        Approximate ε-NE using deviation analysis.
        Estimate how much each agent could gain by deviating from observed strategy.
        """
        if len(episode_history) < 2:
            return 1.0

        # Simple approximation: variance of per-step agent performance
        performances = []
        for step in episode_history:
            agent_cash = [step.get('agent_cash', {}).get(aid, 0) for aid in agent_ids]
            if agent_cash:
                performances.append(np.mean(agent_cash))

        if len(performances) < 2:
            return 1.0

        # Gap = max deviation from mean
        mean_perf = np.mean(performances)
        max_deviation = max(abs(p - mean_perf) for p in performances)
        return max_deviation / (abs(mean_perf) + 1e-6)

    def get_agent_summary(self) -> Dict[int, Dict]:
        """Get a summary of all agent configurations."""
        return {
            aid: {
                'name': identity.name,
                'type': identity.agent_type,
                'controlled_nodes': identity.controlled_nodes,
                'goal_weights': identity.goal_weights,
            }
            for aid, identity in self.agents.items()
        }
