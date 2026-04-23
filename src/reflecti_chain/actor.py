"""
Action Actor (πθ): LLM-driven candidate action generation.
Implements System 1 (intuitive proposal) from the ReflectiChain paper.
"""
import re
import json
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from config import LLMConfig, AgentConfig
from src.semi_sim.environment import Observation, Action


@dataclass
class CandidateAction:
    """A candidate action with metadata."""
    action: Action
    llm_raw_text: str
    confidence: float = 1.0


class ActionActor:
    """
    System 1: Rapid candidate strategy generation using LLM.

    Samples N diverse candidate strategies via high-temperature generation.
    Each candidate is a supply chain intervention described in natural language.
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        agent_config: AgentConfig,
        agent_id: int = 0,
        agent_type: str = "profit_driven",
        llm_client=None,
    ):
        self.cfg = llm_config
        self.agent_cfg = agent_config
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.llm = llm_client  # Can be OpenAI-compatible API client, vLLM, etc.

        # Prompt templates
        self._build_prompts()

    def _build_prompts(self):
        """Construct system prompts for each agent type."""
        if self.agent_type == "profit_driven":
            self.system_prompt = (
                "You are an expert supply chain strategist specializing in profit maximization. "
                "Analyze the current supply chain state and generate strategic interventions. "
                "Prioritize actions that maximize total cash flow while maintaining minimum operational requirements."
            )
        elif self.agent_type == "resilience_driven":
            self.system_prompt = (
                "You are an expert supply chain strategist specializing in system resilience. "
                "Analyze the current supply chain state and generate interventions that maintain "
                "operational continuity under stress. Prioritize risk mitigation and supply security."
            )
        elif self.agent_type == "compliance_driven":
            self.system_prompt = (
                "You are an expert supply chain strategist specializing in regulatory compliance. "
                "Analyze the current supply chain state and generate interventions that strictly adhere "
                "to international trade controls. Compliance is non-negotiable."
            )
        else:
            self.system_prompt = (
                "You are an expert supply chain strategist. Analyze the current supply chain state "
                "and generate strategic interventions balancing profitability, resilience, and compliance."
            )

    def generate_candidates(
        self,
        obs: Observation,
        context: Optional[Dict] = None,
        num_candidates: Optional[int] = None,
    ) -> List[CandidateAction]:
        """
        Generate N candidate actions via high-temperature sampling.

        Args:
            obs: Current POMDP observation
            context: Optional additional context (historical actions, goals)

        Returns:
            List of CandidateAction objects
        """
        num_candidates = num_candidates or self.agent_cfg.num_candidates
        temperature = self.cfg.temperature_action

        # Build strategic prompt
        prompt_text = self._build_strategic_prompt(obs, context)

        candidates = []
        for i in range(num_candidates):
            # Adjust temperature for diversity
            temp = temperature * (1.0 + i * 0.1)

            raw_text = self._call_llm(prompt_text, temperature=temp)
            action = self._parse_llm_response(raw_text, obs, candidate_idx=i)
            candidates.append(CandidateAction(
                action=action,
                llm_raw_text=raw_text,
                confidence=self._estimate_confidence(raw_text),
            ))

        return candidates

    def _build_strategic_prompt(self, obs: Observation, context: Optional[Dict]) -> str:
        """Construct the full strategic reasoning prompt."""
        lines = [
            self.system_prompt,
            "",
            "=== CURRENT SUPPLY CHAIN STATE ===",
            obs.to_text(),
            "",
            "=== INSTRUCTIONS ===",
            "Generate exactly ONE supply chain intervention as a JSON object with the following fields:",
            '  node_id: int (which node to act on, 0-5)',
            '  procurement_volume: float (units to purchase, 0-10000)',
            '  shipment_volume: float (units to ship, 0-5000)',
            '  compliance_shift: float (change in compliance score, -20 to +20)',
            '  target_compliance: float (target compliance level 0-100)',
            '  negotiate_contract: bool',
            '  hedge_risk: bool',
            '  reasoning: str (2-sentence justification)',
            "",
            "Output ONLY the JSON object, no explanation.",
            f"Strategy #{context.get('step', 0) if context else 0 + 1}:",
        ]
        return "\n".join(lines)

    def _call_llm(self, prompt: str, temperature: float = 0.8) -> str:
        """Call the LLM API."""
        if self.llm is not None:
            try:
                response = self.llm.chat.completions.create(
                    model=self.cfg.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=self.cfg.max_tokens_action,
                    top_p=self.cfg.top_p,
                )
                return response.choices[0].message.content
            except Exception as e:
                # Fallback to simulated response
                return self._fallback_response()

        # ── Fallback: rule-based candidate generation ────────────────────────
        return self._fallback_response()

    def _fallback_response(self) -> str:
        """Generate a rule-based candidate when LLM is unavailable."""
        import random
        strategies = [
            {
                "node_id": random.randint(0, 5),
                "procurement_volume": random.uniform(1000, 5000),
                "shipment_volume": random.uniform(500, 3000),
                "compliance_shift": random.uniform(-5, 10),
                "target_compliance": random.uniform(70, 95),
                "negotiate_contract": random.choice([True, False]),
                "hedge_risk": random.choice([True, False]),
                "reasoning": "Strategic supply chain intervention for operational optimization."
            },
            {
                "node_id": random.randint(0, 5),
                "procurement_volume": random.uniform(2000, 8000),
                "shipment_volume": random.uniform(1000, 4000),
                "compliance_shift": random.uniform(5, 15),
                "target_compliance": random.uniform(85, 100),
                "negotiate_contract": True,
                "hedge_risk": random.choice([True, False]),
                "reasoning": "Supply security and compliance enhancement action."
            },
            {
                "node_id": random.randint(0, 5),
                "procurement_volume": random.uniform(500, 2000),
                "shipment_volume": random.uniform(2000, 5000),
                "compliance_shift": random.uniform(-10, 5),
                "target_compliance": random.uniform(60, 80),
                "negotiate_contract": False,
                "hedge_risk": True,
                "reasoning": "Inventory optimization and risk hedging strategy."
            },
        ]
        import json as json_module
        return json_module.dumps(random.choice(strategies))

    def _parse_llm_response(
        self,
        raw_text: str,
        obs: Observation,
        candidate_idx: int,
    ) -> Action:
        """Parse LLM output into an Action object."""
        try:
            # Extract JSON from response
            json_str = self._extract_json(raw_text)
            data = json.loads(json_str)

            return Action(
                agent_id=self.agent_id,
                node_id=int(data.get('node_id', 0)) % obs.node_states.__len__(),
                description=data.get('reasoning', 'Strategic intervention'),
                procurement_volume=float(data.get('procurement_volume', 1000)),
                shipment_volume=float(data.get('shipment_volume', 1000)),
                compliance_shift=float(data.get('compliance_shift', 0)),
                target_compliance=float(data.get('target_compliance', 80)),
                negotiate_contract=bool(data.get('negotiate_contract', False)),
                hedge_risk=bool(data.get('hedge_risk', False)),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback to safe default action
            return Action(
                agent_id=self.agent_id,
                node_id=candidate_idx % len(obs.node_states),
                description="Default strategic action",
                procurement_volume=2000.0,
                shipment_volume=1500.0,
                compliance_shift=0.0,
                target_compliance=80.0,
                negotiate_contract=False,
                hedge_risk=True,
            )

    def _extract_json(self, text: str) -> str:
        """Extract JSON object from LLM response text."""
        # Try to find JSON object
        text = text.strip()
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end > start:
            return text[start:end]
        # Fallback
        return text

    def _estimate_confidence(self, raw_text: str) -> float:
        """Estimate LLM's confidence in its response."""
        text_lower = raw_text.lower()
        uncertainty_words = ['maybe', 'perhaps', 'might', 'could be', 'uncertain', 'unclear']
        count = sum(1 for w in uncertainty_words if w in text_lower)
        return max(0.3, 1.0 - count * 0.1)
