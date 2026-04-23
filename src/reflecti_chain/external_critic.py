"""
External Critic (Vϕe): Post-execution assessment and retrospective evaluation.
Implements Reflection-on-Action for hindsight-driven credit assignment.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from config import LLMConfig
from src.semi_sim.environment import Observation


@dataclass
class ExecutionResult:
    """Result of executing an action in the environment."""
    obs: Observation
    rewards: Dict[str, float]
    info: Dict[str, Any]
    obs_next: Optional[Observation] = None


@dataclass
class HindsightRecord:
    """Record for retrospective evaluation."""
    timestep: int
    obs: Observation
    action: Any
    immediate_reward: float
    execution_feedback: Dict[str, Any]


class ExternalCritic:
    """
    External Reflection: post-execution assessment and retrospective evaluation.

    Responsibilities:
        1. Immediate Assessment: evaluate execution result at each step
        2. Retrospective Hindsight: re-evaluate historical actions with future context
        3. Long-horizon credit assignment: propagate future outcomes back to past decisions
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        memory_window: int = 3,
        llm_client=None,
    ):
        self.cfg = llm_config
        self.memory_window = memory_window
        self.llm = llm_client

        # Working memory buffer (Algorithm 1, line 14-15)
        self.memory_buffer: List[HindsightRecord] = []
        self.episode_buffer: List[HindsightRecord] = []

    def immediate_assessment(
        self,
        obs: Observation,
        action: Any,
        execution_result: ExecutionResult,
    ) -> float:
        """
        Phase 2 (Algorithm 1, line 14): Immediate assessment after execution.
        Returns f^e_t ∈ ℝ (immediate reward signal).
        """
        rewards = execution_result.rewards

        # Combine multi-dimensional rewards
        immediate_score = (
            0.3 * rewards.get('cash', 0) +
            0.3 * rewards.get('compliance', 0) +
            0.2 * rewards.get('risk', 0) +
            0.2 * rewards.get('bullwhip', 0)
        )

        # Record in working memory
        self.record(
            timestep=obs.timestep,
            obs=obs,
            action=action,
            immediate_reward=immediate_score,
            execution_feedback=rewards,
        )

        return immediate_score

    def record(
        self,
        timestep: int,
        obs: Observation,
        action: Any,
        immediate_reward: float,
        execution_feedback: Dict[str, Any],
    ):
        """Store a record in the working memory buffer."""
        record = HindsightRecord(
            timestep=timestep,
            obs=obs,
            action=action,
            immediate_reward=immediate_reward,
            execution_feedback=execution_feedback,
        )
        self.memory_buffer.append(record)
        self.episode_buffer.append(record)

    def retrospective_evaluation(
        self,
        hindsight_buffer: Optional[List[HindsightRecord]] = None,
    ) -> Dict[int, float]:
        """
        Phase 2 (Algorithm 1, line 19-20): Retrospective hindsight evaluation.

        Re-evaluates historical actions within the memory window K,
        using full hindsight context to correct early false-positive evaluations.

        Returns:
            Dict[timestep -> retrospective_score s_retro]
        """
        buffer = hindsight_buffer or self.memory_buffer
        if len(buffer) == 0:
            return {}

        K = min(self.memory_window, len(buffer))
        recent_buffer = buffer[-K:]

        retrospective_scores = {}

        for i, record in enumerate(recent_buffer):
            # Future context: look at outcomes in subsequent steps
            future_context = self._extract_future_context(buffer, record.timestep)

            # Retrospective score computation
            s_retro = self._compute_retrospective_score(record, future_context, buffer)
            retrospective_scores[record.timestep] = s_retro

        return retrospective_scores

    def _extract_future_context(
        self,
        buffer: List[HindsightRecord],
        reference_timestep: int,
    ) -> Dict[str, Any]:
        """Extract future outcomes after a given timestep for hindsight analysis."""
        future_records = [r for r in buffer if r.timestep > reference_timestep]

        if not future_records:
            return {'future_rewards': [], 'avg_future_risk': 0.5}

        future_rewards = [r.immediate_reward for r in future_records]
        future_risks = []
        for r in future_records:
            if hasattr(r.obs, 'global_risk'):
                future_risks.append(r.obs.global_risk)

        return {
            'future_rewards': future_rewards,
            'avg_future_risk': np.mean(future_risks) if future_risks else 0.5,
            'future_length': len(future_records),
            'cumulative_future_reward': sum(future_rewards),
        }

    def _compute_retrospective_score(
        self,
        record: HindsightRecord,
        future_context: Dict[str, Any],
        buffer: List[HindsightRecord],
    ) -> float:
        """
        Compute retrospective score s_retro ∈ [0, 100].

        This corrects early false positives by observing materialized cascade effects:
        - Immediate positive + negative future = false positive (short-term thinking)
        - Immediate negative + positive future = false negative (long-term investment)
        - Consistent signs = accurate evaluation
        """
        immediate = record.immediate_reward

        # Future reward context
        future_rewards = future_context.get('future_rewards', [])
        future_risk = future_context.get('avg_future_risk', 0.5)

        if future_rewards:
            avg_future = np.mean(future_rewards)
            cumulative_future = sum(future_rewards)
        else:
            avg_future = 0.0
            cumulative_future = 0.0

        # Hindsight correction logic
        # If immediate was good but future got worse -> downgrade (overconfidence)
        if immediate > 0 and cumulative_future < -0.1:
            hindsight_correction = -0.2  # Penalize false positive
        # If immediate was bad but future recovered -> upgrade (worth it)
        elif immediate < 0 and cumulative_future > 0.1:
            hindsight_correction = +0.15  # Reward patience
        else:
            hindsight_correction = 0.0

        # Risk-aware adjustment
        if future_risk > 0.7:
            hindsight_correction -= 0.1  # High future risk penalizes current action
        elif future_risk < 0.3:
            hindsight_correction += 0.05

        # Combine: retrospective score = immediate + correction
        # Scale to [0, 100]
        raw_score = immediate + hindsight_correction
        scaled_score = (raw_score + 1.0) * 50.0  # [-1,1] -> [0, 100]

        return max(0.0, min(100.0, scaled_score))

    def compute_policy_gradient_signals(
        self,
        hindsight_scores: Dict[int, float],
    ) -> Dict[int, float]:
        """
        Convert retrospective scores to policy gradient signals (Algorithm 1, line 20).

        Maps s_retro ∈ [0, 100] to normalized reward r ∈ [-1, 1]:

            r = 2 * (s_retro / 100) - 1
        """
        policy_signals = {}
        for timestep, s_retro in hindsight_scores.items():
            r = 2.0 * (s_retro / 100.0) - 1.0
            policy_signals[timestep] = r

        return policy_signals

    def clear_buffer(self):
        """Clear the working memory buffer after policy update."""
        self.memory_buffer = []

    def clear_episode(self):
        """Clear the full episode buffer."""
        self.episode_buffer = []
        self.memory_buffer = []

    def get_buffer(self) -> List[HindsightRecord]:
        return list(self.memory_buffer)

    def get_episode_buffer(self) -> List[HindsightRecord]:
        return list(self.episode_buffer)
