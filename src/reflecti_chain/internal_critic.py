"""
Internal Critic (Vϕi): Semantic compliance scoring.
Implements Reflection-in-Action (System 2) — LLM-based semantic evaluation.
"""
import json
from typing import List, Dict, Optional
from config import LLMConfig
from src.semi_sim.environment import Observation, Action


class InternalCritic:
    """
    System 2 (Internal Reflection): LLM-based semantic compliance evaluation.

    Evaluates candidate actions for policy text compliance and strategic coherence.
    Assigns semantic score s_llm ∈ [0, 100].
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        agent_type: str = "profit_driven",
        llm_client=None,
    ):
        self.cfg = llm_config
        self.agent_type = agent_type
        self.llm = llm_client

    def score_action(
        self,
        obs: Observation,
        action,
        candidate_text: str,
    ) -> float:
        """
        Evaluate a candidate action for semantic compliance.

        Returns:
            s_llm: semantic compliance score in [0, 100]
        """
        if self.llm is not None:
            try:
                return self._llm_score(obs, action, candidate_text)
            except Exception:
                pass

        # Fallback: rule-based scoring
        return self._rule_based_score(obs, action)

    def score_batch(
        self,
        obs: Observation,
        candidates: List,
    ) -> List[float]:
        """Score all candidates and return list of semantic scores."""
        return [self.score_action(obs, c.action, c.llm_raw_text) for c in candidates]

    def _llm_score(
        self,
        obs: Observation,
        action,
        candidate_text: str,
    ) -> float:
        """Use LLM for semantic scoring."""
        prompt = self._build_scoring_prompt(obs, action, candidate_text)

        try:
            response = self.llm.chat.completions.create(
                model=self.cfg.model_name,
                messages=[
                    {"role": "system", "content": "You are a supply chain compliance evaluator. Score actions 0-100."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.cfg.temperature_reflect,
                max_tokens=self.cfg.max_tokens_reflect,
            )
            text = response.choices[0].message.content
            return self._parse_score(text)
        except Exception:
            return self._rule_based_score(obs, action)

    def _build_scoring_prompt(
        self,
        obs: Observation,
        action,
        candidate_text: str,
    ) -> str:
        """Build the scoring prompt."""
        state_summary = f"""
=== Current State ===
{obs.to_text()}

=== Candidate Action ===
{candidate_text}

=== Task ===
Evaluate this candidate action on the following criteria (0-100 scale):
1. Policy Compliance: Does it respect international trade controls and sanctions?
2. Strategic Coherence: Is it logically consistent with current supply chain conditions?
3. Risk Appropriateness: Does it appropriately account for current risk levels?

Provide a single overall score (0-100) where:
  0-30: High risk / non-compliant
  31-60: Moderate risk / partially compliant
  61-80: Good / mostly compliant
  81-100: Excellent / fully compliant

Output ONLY the numeric score, nothing else.
"""
        return state_summary

    def _parse_score(self, text: str) -> float:
        """Extract numeric score from LLM response."""
        import re
        text = text.strip()
        # Look for numbers
        matches = re.findall(r'\d+(?:\.\d+)?', text)
        if matches:
            score = float(matches[0])
            return max(0.0, min(100.0, score))
        return 60.0  # Default moderate score

    def _rule_based_score(self, obs: Observation, action) -> float:
        """Rule-based fallback scoring."""
        score = 70.0  # Start with moderate score

        # Compliance check
        if action.target_compliance >= 85:
            score += 15
        elif action.target_compliance >= 70:
            score += 5
        elif action.target_compliance < 50:
            score -= 20

        # Risk hedging bonus
        if action.hedge_risk:
            score += 8

        # Contract negotiation bonus
        if action.negotiate_contract:
            score += 5

        # Current global risk penalty
        if obs.global_risk > 0.6:
            score -= 10
        elif obs.global_risk < 0.3:
            score += 5

        # Active shock penalty
        if obs.active_shock:
            if 'ban' in obs.active_shock or 'embargo' in obs.active_shock:
                score -= 15
            else:
                score -= 8

        return max(0.0, min(100.0, score))
