"""
ReflectiChain Agent: Unified cognitive agentic framework.
Integrates all components: Action Actor, Internal/External Critics, World Model, LoRA updater.
Implements Algorithm 1 from the paper.
"""
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

from config import AgentConfig, LLMConfig, WorldModelConfig, SemiSimConfig
from src.semi_sim.enhanced_env import EnhancedSemiSimEnvironment as SemiSimEnvironment, Observation, Action
from src.semi_sim.world_model import WorldModelWrapper
from .actor import ActionActor, CandidateAction
from .internal_critic import InternalCritic
from .external_critic import ExternalCritic, HindsightRecord, ExecutionResult
from .lora_update import LoRAUpdater


@dataclass
class DecisionRecord:
    """Record of a single decision-making step."""
    timestep: int
    candidates: List[CandidateAction]
    semantic_scores: List[float]
    wm_scores: List[float]
    joint_scores: List[float]
    selected_action: Optional[Action]
    immediate_reward: float
    retrospective_score: float = 0.0
    policy_gradient: float = 0.0


class ReflectiChainAgent:
    """
    ReflectiChain: Cognitive Agentic Framework for Resilient Supply Chain Planning.

    Implements the full Algorithm 1:
        Phase 1: Reflection-in-Action (Dual-System Latent Rehearsal)
            - System 1: Candidate generation via πθ
            - System 2: Joint evaluation via Vϕi (semantic) + SC-WM (physical)
        Phase 2: Reflection-on-Action (Double-Loop Hindsight Learning)
            - Immediate assessment via Vϕe
            - Retrospective evaluation with future context
            - Policy gradient update via Test-time LoRA
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        agent_config: AgentConfig,
        world_model_cfg: WorldModelConfig,
        semi_cfg: SemiSimConfig,
        agent_id: int = 0,
        agent_type: str = "profit_driven",
        llm_client=None,
        device: str = "cuda",
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.cfg = agent_config
        self.device = device

        # ── Component initialization ─────────────────────────────────────────
        self.actor = ActionActor(
            llm_config=llm_config,
            agent_config=agent_config,
            agent_id=agent_id,
            agent_type=agent_type,
            llm_client=llm_client,
        )

        self.internal_critic = InternalCritic(
            llm_config=llm_config,
            agent_type=agent_type,
            llm_client=llm_client,
        )

        self.external_critic = ExternalCritic(
            llm_config=llm_config,
            memory_window=agent_config.memory_window,
            llm_client=llm_client,
        )

        self.lora_updater = LoRAUpdater(
            agent_config=agent_config,
            llm_config=llm_config,
            device=device,
        )

        self.world_model = WorldModelWrapper(
            cfg=world_model_cfg,
            semi_cfg=semi_cfg,
            device=device,
        )

        # ── Decision records ─────────────────────────────────────────────────
        self.decision_history: List[DecisionRecord] = []
        self.episode_reward_history: List[float] = []

        # ── State ────────────────────────────────────────────────────────────
        self.step_count = 0
        self.lora_initialized = False

    def _init_lora(self, base_model=None):
        """Initialize LoRA adaptation layer."""
        if not self.lora_initialized:
            self.lora_updater.initialize(base_model)
            self.lora_initialized = True

    # ─── Phase 1: Reflection-in-Action ─────────────────────────────────────────
    def reflection_in_action(
        self,
        obs: Observation,
    ) -> Tuple[Action, List[float], List[float], List[float]]:
        """
        Phase 1: Dual-System Latent Rehearsal.

        Steps:
            1. Sample N candidate strategies via πθ (System 1)
            2. Evaluate each candidate:
                - Vϕi: semantic compliance score
                - SC-WM: physical grounding reward
            3. Select optimal action via joint score

        Returns:
            selected_action, semantic_scores, wm_scores, joint_scores
        """
        # Step 1: Generate candidates (System 1)
        candidates = self.actor.generate_candidates(
            obs, context={'step': self.step_count}
        )

        # Step 2: Joint evaluation (System 2)
        semantic_scores = []
        wm_scores = []
        joint_scores = []

        for cand in candidates:
            # Semantic evaluation
            s_llm = self.internal_critic.score_action(
                obs, cand.action, cand.llm_raw_text
            )
            semantic_scores.append(s_llm)

            # Physical grounding via World Model
            r_wm = self._eval_via_world_model(obs, cand.action)
            wm_scores.append(r_wm)

            # Joint score: J(k) = α·s_llm + β·r_wm
            J = self.cfg.alpha_semantic * (s_llm / 100.0) + self.cfg.beta_physical * r_wm
            joint_scores.append(J)

        # Step 3: Select optimal action
        best_idx = int(np.argmax(joint_scores))
        selected_action = candidates[best_idx].action

        # Record decision
        self.decision_history.append(DecisionRecord(
            timestep=self.step_count,
            candidates=candidates,
            semantic_scores=semantic_scores,
            wm_scores=wm_scores,
            joint_scores=joint_scores,
            selected_action=selected_action,
            immediate_reward=0.0,  # Will be updated in Phase 2
        ))

        return selected_action, semantic_scores, wm_scores, joint_scores

    def _eval_via_world_model(
        self,
        obs: Observation,
        action,
    ) -> float:
        """Evaluate action via World Model physical grounding."""
        try:
            r_wm, _ = self.world_model.latent_rollout(
                current_node_features=np.concatenate([
                    s.to_vector() for s in obs.node_states.values()
                ]),
                current_edge_features=self._get_edge_features(obs),
                action_encoding=self._action_to_encoding(action),
                num_steps=self.world_model.cfg.trajectory_length,
            )
            return float(r_wm)
        except Exception:
            # Fallback: use simple heuristic
            return self._heuristic_grounding(obs, action)

    def _heuristic_grounding(self, obs: Observation, action) -> float:
        """Fallback heuristic when World Model is unavailable."""
        score = 0.0

        # Inventory balance
        avg_inv = np.mean([s.inventory for s in obs.node_states.values()])
        if 2000 < avg_inv < 8000:
            score += 0.2

        # Cash adequacy
        total_cash = sum(s.cash for s in obs.node_states.values())
        if total_cash > 300000:
            score += 0.2

        # Compliance
        avg_comp = np.mean([s.compliance for s in obs.node_states.values()])
        if avg_comp > 70:
            score += 0.2

        # Risk level
        if obs.global_risk < 0.5:
            score += 0.2

        # Action-specific
        if action.hedge_risk:
            score += 0.1
        if action.negotiate_contract:
            score += 0.1

        return max(-1.0, min(1.0, score - 0.5))

    def _get_edge_features(self, obs: Observation) -> np.ndarray:
        """Extract edge features from observation."""
        edge_features = []
        G = obs.graph_structure
        if G is None:
            return np.zeros(14, dtype=np.float32)  # 7 edges * 2 features
        for u, v in G.edges():
            w = G.edges[u, v].get('weight', 0.3)
            flow = G.edges[u, v].get('current_flow', 0)
            edge_features.extend([w, flow])
        return np.array(edge_features, dtype=np.float32)

    def _action_to_encoding(self, action) -> np.ndarray:
        """Convert action to World Model encoding."""
        return np.array([
            action.procurement_volume / 10000.0,
            action.shipment_volume / 10000.0,
            action.compliance_shift / 100.0,
            action.target_compliance / 100.0,
        ], dtype=np.float32)

    # ─── Phase 2: Reflection-on-Action ────────────────────────────────────────
    def reflection_on_action(
        self,
        obs: Observation,
        executed_action: Action,
        execution_result: ExecutionResult,
    ) -> Tuple[float, float, float]:
        """
        Phase 2: Double-Loop Hindsight Learning.

        Steps:
            1. Immediate assessment: f^e_t via Vϕe
            2. Retrospective evaluation with future context
            3. Compute policy gradient signal
            4. Trigger LoRA update if buffer is full

        Returns:
            immediate_reward, retrospective_score, policy_gradient
        """
        # Step 1: Immediate assessment
        immediate_reward = self.external_critic.immediate_assessment(
            obs, executed_action, execution_result
        )

        # Update decision record
        if self.decision_history:
            self.decision_history[-1].immediate_reward = immediate_reward

        self.episode_reward_history.append(immediate_reward)

        # Add to LoRA training buffer
        self.lora_updater.add_gradient_sample(
            obs_text=obs.to_text(),
            action_text=executed_action.description,
            reward=immediate_reward,
        )

        # Step 2-4: Retrospective + Policy Gradient (only when buffer full)
        retrospective_score = 0.0
        policy_gradient = 0.0

        if self.lora_updater.should_update():
            # Retrospective evaluation
            hindsight_scores = self.external_critic.retrospective_evaluation()

            if hindsight_scores:
                # Use the most recent retrospective score
                latest_t = max(hindsight_scores.keys())
                retrospective_score = hindsight_scores[latest_t]

                # Compute policy gradient signals
                policy_signals = self.external_critic.compute_policy_gradient_signals(
                    hindsight_scores
                )

                # Update decision records with retrospective info
                for t, pg in policy_signals.items():
                    for rec in reversed(self.decision_history):
                        if rec.timestep == t:
                            rec.retrospective_score = hindsight_scores.get(t, 0)
                            rec.policy_gradient = pg
                            break

                # Update LoRA
                loss = self.lora_updater.update()

                # Clear hindsight buffer
                self.external_critic.clear_buffer()

        return immediate_reward, retrospective_score, policy_gradient

    # ─── Main Step ─────────────────────────────────────────────────────────────
    def step(
        self,
        obs: Observation,
        env=None,
    ) -> Tuple[Action, Dict]:
        """
        Execute one full decision step (Algorithm 1).

        Returns:
            selected_action, info dict with all scores
        """
        self.step_count += 1

        # Phase 1: Reflection-in-Action
        action, sem_scores, wm_scores, joint_scores = self.reflection_in_action(obs)

        # Execute in environment
        if env is not None:
            obs_next, rewards, done, info = env.step([action])
            exec_result = ExecutionResult(
                obs=obs, rewards=rewards, info=info, obs_next=obs_next
            )

            # Phase 2: Reflection-on-Action
            imm_r, retro_s, pg = self.reflection_on_action(
                obs, action, exec_result
            )
        else:
            obs_next = obs
            rewards = {'total': 0.0}
            done = False
            info = {}
            imm_r, retro_s, pg = 0.0, 0.0, 0.0

        info.update({
            'step': self.step_count,
            'semantic_scores': sem_scores,
            'wm_scores': wm_scores,
            'joint_scores': joint_scores,
            'immediate_reward': imm_r,
            'retrospective_score': retro_s,
            'policy_gradient': pg,
            'lora_stats': self.lora_updater.get_stats(),
        })

        return action, info

    def reset(self):
        """Reset agent state for a new episode."""
        self.step_count = 0
        self.decision_history = []
        self.episode_reward_history = []
        self.external_critic.clear_episode()
        self.lora_updater.training_buffer = []

    # ─── Analysis ──────────────────────────────────────────────────────────────
    def get_episode_summary(self) -> Dict[str, float]:
        """Get summary statistics for the current episode."""
        if not self.episode_reward_history:
            return {}

        rewards = np.array(self.episode_reward_history)
        return {
            'total_steps': self.step_count,
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'cumulative_reward': float(np.sum(rewards)),
            'positive_ratio': float(np.mean(rewards > 0)),
        }

    def get_decision_trace(self) -> List[Dict]:
        """Get detailed trace of all decisions made."""
        return [
            {
                'timestep': rec.timestep,
                'num_candidates': len(rec.candidates),
                'best_semantic': max(rec.semantic_scores) if rec.semantic_scores else 0,
                'best_wm': max(rec.wm_scores) if rec.wm_scores else 0,
                'best_joint': max(rec.joint_scores) if rec.joint_scores else 0,
                'immediate_reward': rec.immediate_reward,
                'retro_score': rec.retrospective_score,
                'policy_gradient': rec.policy_gradient,
            }
            for rec in self.decision_history
        ]
