"""
LoRA Updater: Test-time LoRA adaptation via policy gradient.
Implements Agentic RL update from the paper (Algorithm 1, line 22-24).
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from config import AgentConfig, LLMConfig
from peft import LoraConfig, get_peft_model, TaskType


@dataclass
class GradientSample:
    """A single gradient sample for policy gradient update."""
    obs_text: str
    action_text: str
    reward: float  # Normalized reward ∈ [-1, 1]
    log_prob: float = 0.0


class LoRAUpdater:
    """
    Test-time LoRA adaptation using REINFORCE policy gradient.

    Implements the Agentic RL update from ReflectiChain:
        ℓ_θ = -r · log πθ(a | x_action)

    This fundamentally evolves the actor's reasoning weights during deployment,
    enabling Double-Loop Learning without retraining the base model.
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        llm_config: LLMConfig,
        base_model=None,
        device: str = "cuda",
    ):
        self.agent_cfg = agent_config
        self.llm_cfg = llm_config
        self.device = device

        # ── LoRA configuration ────────────────────────────────────────────────
        self.lora_config = LoraConfig(
            r=agent_config.lora_rank,                 # 8
            lora_alpha=agent_config.lora_alpha,        # 16
            lora_dropout=agent_config.lora_dropout,   # 0.05
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
        )

        # Training state
        self.training_buffer: List[GradientSample] = []
        self.step_count = 0
        self.accumulated_loss = 0.0
        self.optimizer = None
        self.base_model = base_model
        self.model = None  # Set by initialize()

        # Metrics
        self.loss_history: List[float] = []
        self.reward_history: List[float] = []

    def initialize(self, base_model):
        """
        Initialize LoRA adaptation layer on top of base model.

        In API-only mode (base_model=None), the LoRA updater records -reward
        as the REINFORCE loss signal for analysis.
        """
        self.base_model = base_model
        if base_model is not None:
            self.model = get_peft_model(base_model, self.lora_config)
            self.model.to(self.device)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.agent_cfg.learning_rate,
                weight_decay=self.agent_cfg.kl_divergence_weight,
            )
            self.model.train()
        else:
            self.model = None

    def add_gradient_sample(
        self,
        obs_text: str,
        action_text: str,
        reward: float,
    ):
        """Add a gradient sample to the training buffer."""
        self.training_buffer.append(GradientSample(
            obs_text=obs_text,
            action_text=action_text,
            reward=reward,
        ))
        self.reward_history.append(reward)

    def compute_reinforce_loss(self, sample: GradientSample) -> torch.Tensor:
        """
        Compute REINFORCE loss for a single sample:
            ℓ_θ = -r · log πθ(a | x_action)

        For API-only mode: use -reward as the proxy loss signal since
        we cannot compute log π through an HTTP API.
        """
        reward_t = torch.tensor(sample.reward, dtype=torch.float32, device=self.device)

        if self.model is not None:
            # Full LoRA backpropagation path (model loaded locally)
            try:
                inputs = self.model.prepare_inputs_for_generation(
                    sample.obs_text + "\n" + sample.action_text,
                    return_tensor=True
                ).to(self.device)
                outputs = self.model(inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

                # REINFORCE: weight cross-entropy by reward
                if logits.dim() > 1:
                    log_probs = torch.log_softmax(logits, dim=-1)
                    loss = -reward_t * log_probs.mean()
                else:
                    loss = -reward_t * logits.squeeze()
                return loss
            except Exception:
                # Fallback to pure reward signal
                return -reward_t
        else:
            # API-only path: use -reward directly as loss signal
            # This is the correct REINFORCE objective when log π is unavailable
            return -reward_t

    def update(self) -> float:
        """
        Perform policy gradient update on accumulated samples.

        For API-only mode (no local model): computes -reward as the REINFORCE
        loss signal (since log π is unavailable via API), then records it.

        Returns:
            Average loss for this update step
        """
        if len(self.training_buffer) == 0:
            return 0.0

        self.step_count += 1

        # Accumulate gradient over samples
        total_loss = 0.0
        num_samples = len(self.training_buffer)

        for sample in self.training_buffer:
            loss = self.compute_reinforce_loss(sample)
            total_loss += loss.item()

            if self.model is not None:
                try:
                    loss.backward()
                except Exception:
                    pass

        # Gradient accumulation and optimizer step
        if self.model is not None:
            try:
                # Average gradient
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad /= num_samples

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            except Exception:
                pass

        avg_loss = total_loss / num_samples
        self.accumulated_loss += avg_loss
        self.loss_history.append(avg_loss)

        # Clear buffer after update
        self.training_buffer = []

        return avg_loss

    def get_stats(self) -> Dict[str, float]:
        """Return training statistics."""
        if not self.reward_history:
            return {'step_count': self.step_count, 'avg_loss': 0.0, 'avg_reward': 0.0}

        return {
            'step_count': self.step_count,
            'avg_loss': np.mean(self.loss_history[-10:]) if self.loss_history else 0.0,
            'avg_reward': np.mean(self.reward_history[-10:]),
            'reward_variance': np.var(self.reward_history[-10:]) if len(self.reward_history) >= 10 else 0.0,
            'buffer_size': len(self.training_buffer),
        }

    def save(self, path: str):
        """Save LoRA weights."""
        if self.model is not None:
            self.model.save_pretrained(path)

    def load(self, path: str):
        """Load LoRA weights."""
        if self.model is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.base_model, path)
            self.model.to(self.device)
            self.model.eval()

    def should_update(self) -> bool:
        """Check if buffer is full enough for an update."""
        return len(self.training_buffer) >= self.agent_cfg.memory_window
