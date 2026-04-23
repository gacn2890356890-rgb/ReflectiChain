"""
Evaluation Metrics: all metrics for Table 1 and beyond.
"""
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class EvalResult:
    """Container for evaluation results."""
    CEE: float       # Total Cash
    RCI: float       # Average Compliance
    BWI: float       # Bullwhip Index
    OR: float        # Operability Ratio (%)
    ARL: float       # Average Risk Level
    step_rewards: List[float]
    cumulative_reward: float

    def to_dict(self) -> Dict[str, float]:
        return {
            'CEE': self.CEE,
            'RCI': self.RCI,
            'BWI': self.BWI,
            'OR': self.OR * 100,  # Convert to percentage
            'ARL': self.ARL,
            'cumulative_reward': self.cumulative_reward,
        }


@dataclass
class GameEvalResult:
    """Container for multi-agent game-theoretic results."""
    SW: float         # Social Welfare
    epsilon_NE: float  # Nash Equilibrium approximation
    CI: float         # Competition Intensity
    CR: float         # Coordination Rate
    RRG: float        # Relative Revenue Gap
    per_agent_metrics: Dict[int, Dict[str, float]]


class MetricCalculator:
    """
    Unified metric computation for ReflectiChain evaluation.

    Handles:
        - Standard metrics (Table 1)
        - Game-theoretic metrics (Multi-Agent)
        - Adversarial metrics (Regret, WCR, Recovery Time)
        - Ablation analysis
    """

    def __init__(self):
        self.episode_rewards: List[float] = []
        self.step_rewards: List[Dict[str, float]] = []

    def compute_standard_metrics(
        self,
        env,
        step_rewards: Optional[List[Dict[str, float]]] = None,
    ) -> EvalResult:
        """Compute standard Table 1 metrics from environment state."""
        metrics = env.compute_metrics()

        # Compute cumulative reward
        if step_rewards:
            total_reward = sum(sr.get('total', 0) for sr in step_rewards)
        else:
            total_reward = 0.0

        return EvalResult(
            CEE=metrics['CEE'],
            RCI=metrics['RCI'],
            BWI=metrics['BWI'],
            OR=metrics['OR'],
            ARL=metrics['ARL'],
            step_rewards=[sr.get('total', 0) for sr in (step_rewards or [])],
            cumulative_reward=total_reward,
        )

    def compute_game_metrics(
        self,
        episode_history: List[Dict],
        agent_ids: List[int],
        num_agents: int,
    ) -> GameEvalResult:
        """Compute multi-agent game-theoretic metrics."""
        # Social Welfare: sum of all agent cumulative cash
        sw = 0.0
        per_agent_cash: Dict[int, float] = {aid: 0.0 for aid in agent_ids}

        for step in episode_history:
            for aid in agent_ids:
                cash = step.get('agent_cash', {}).get(aid, 0)
                per_agent_cash[aid] += cash
                sw += cash

        # Competition Intensity: std of agent performances
        if len(agent_ids) > 1:
            performances = list(per_agent_cash.values())
            ci = float(np.std(performances))
        else:
            ci = 0.0

        # Nash Equilibrium approximation (ε-NE)
        epsilon_ne = self._compute_nash_gap(per_agent_cash)

        # Coordination Rate: fraction of steps with successful cooperation
        coop_steps = sum(
            1 for step in episode_history
            if step.get('cooperation_success', False)
        )
        cr = coop_steps / max(len(episode_history), 1)

        # Relative Revenue Gap: gap between best and worst agent
        cash_values = list(per_agent_cash.values())
        rrg = (max(cash_values) - min(cash_values)) / (abs(max(cash_values)) + 1e-6)

        per_agent_metrics = {}
        for aid in agent_ids:
            per_agent_metrics[aid] = {
                'cumulative_cash': per_agent_cash[aid],
                'avg_compliance': np.mean([
                    s.get(f'agent_compliance_{aid}', 80.0)
                    for s in episode_history
                ]),
                'avg_risk': np.mean([
                    s.get(f'agent_risk_{aid}', 0.3)
                    for s in episode_history
                ]),
            }

        return GameEvalResult(
            SW=sw,
            epsilon_NE=epsilon_ne,
            CI=ci,
            CR=cr,
            RRG=rrg,
            per_agent_metrics=per_agent_metrics,
        )

    def _compute_nash_gap(self, per_agent_cash: Dict[int, float]) -> float:
        """
        Approximate ε-NE: estimate how much each agent could gain by unilateral deviation.
        """
        agents = list(per_agent_cash.keys())
        if len(agents) < 2:
            return 0.0

        cash_values = np.array(list(per_agent_cash.values()))
        mean_cash = np.mean(cash_values)
        max_deviation = np.max(np.abs(cash_values - mean_cash))

        if abs(mean_cash) < 1e-6:
            return 1.0

        return float(max_deviation / abs(mean_cash))

    def compute_adversarial_metrics(
        self,
        episode_rewards: List[float],
        oracle_rewards: List[float],
        recovery_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Compute adversarial evaluation metrics.

        Args:
            episode_rewards: Actual rewards under adversarial shock
            oracle_rewards: Optimal rewards (from perfect agent)
            recovery_steps: Steps to recover from OR < 30% to OR > 80%
        """
        actual_cum = np.sum(episode_rewards)
        oracle_cum = np.sum(oracle_rewards)

        # Regret = Oracle - Actual
        regret = oracle_cum - actual_cum
        normalized_regret = regret / (abs(oracle_cum) + 1e-6)

        # Worst-case survival: fraction of episodes with OR > threshold
        # (simplified: check if average risk stayed below threshold)
        avg_risk = np.mean([1 - r for r in episode_rewards])
        survival = 1.0 if avg_risk < 0.5 else 0.0

        return {
            'Regret': float(regret),
            'NormalizedRegret': float(normalized_regret),
            'CumulativeActual': float(actual_cum),
            'CumulativeOracle': float(oracle_cum),
            'WCR': float(survival),
            'RecoverySteps': float(recovery_steps or 0),
        }

    def compute_ablation_metrics(
        self,
        variant_name: str,
        step_rewards: List[float],
        retro_scores: List[float],
        failure_mode: str = "",
    ) -> Dict[str, Any]:
        """Compute ablation study metrics."""
        avg_step_reward = np.mean(step_rewards) if step_rewards else 0.0
        avg_retro = np.mean(retro_scores) if retro_scores else 0.0

        return {
            'variant': variant_name,
            'avg_step_reward': float(avg_step_reward),
            'avg_retro_score': float(avg_retro),
            'total_cumulative': float(np.sum(step_rewards)) if step_rewards else 0.0,
            'failure_mode': failure_mode,
        }

    def compare_with_baseline(
        self,
        reflecti_metrics: EvalResult,
        baselines: Dict[str, EvalResult],
    ) -> Dict[str, Dict[str, float]]:
        """Compare ReflectiChain against all baselines."""
        results = {}
        for name, base in baselines.items():
            results[name] = {
                'CEE_improvement': reflecti_metrics.CEE - base.CEE,
                'OR_improvement': (reflecti_metrics.OR - base.OR) * 100,
                'ARL_improvement': base.ARL - reflecti_metrics.ARL,  # Lower is better
                'RCI_gap': reflecti_metrics.RCI - base.RCI,
            }
        return results

    def print_table1(
        self,
        results: Dict[str, EvalResult],
        ideal: Dict[str, float],
    ):
        """Print formatted Table 1 as in the paper."""
        header = f"{'Metric':<25} {'PPO':>12} {'Qwen':>12} {'InternLM':>12} {'ReflectiChain':>14} {'Ideal Ref.':>12}"
        separator = "-" * len(header)

        rows = [
            ('Total Cash (CEE)', 'CEE', 0),
            ('Avg Compliance (RCI)', 'RCI', 1),
            ('Bullwhip Index (BWI)', 'BWI', 2),
            ('Operability Ratio (OR)', 'OR', 3),
            ('Average Risk Level (ARL)', 'ARL', 4),
        ]

        print("\n" + "=" * 95)
        print(f"{'Table 1: Metric-level comparison under macroeconomic shocks':^95}")
        print("=" * 95)
        print(header)
        print(separator)

        for display_name, key, dec_idx in rows:
            values = []
            for name in ['PPO', 'Qwen2.5-7B', 'InternLM2.5', 'ReflectiChain']:
                res = results.get(name)
                if res is None:
                    values.append('N/A')
                    continue
                val = getattr(res, key)
                if dec_idx == 0:
                    values.append(f'{val:,.0f}')
                elif dec_idx == 3:
                    values.append(f'{val*100:.1f}%')
                else:
                    values.append(f'{val:.2f}')

            ideal_val = ideal.get(key, '> 0')
            print(f"{display_name:<25} {values[0]:>12} {values[1]:>12} {values[2]:>12} {values[3]:>14} {str(ideal_val):>12}")

        print(separator)
        print()
