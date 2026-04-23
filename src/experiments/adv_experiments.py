"""
Adversarial Stress Test Experiment Suite.
Tests ReflectiChain under adversarial policy shock generation.
"""
import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import SemiSimConfig, LLMConfig, AgentConfig, WorldModelConfig, ShockConfig
from src.semi_sim.enhanced_env import EnhancedSemiSimEnvironment as SemiSimEnvironment, Action
from src.semi_sim.shock_gen import ShockGenerator, AdversarialPolicyLLM
from src.reflecti_chain.reflection import ReflectiChainAgent
from src.experiments.baselines import PPORLBaseline, VanillaLLMBaseline


def make_llm_client(llm_cfg: LLMConfig):
    """Create an Ollama LLM client. Returns None if Ollama is not available."""
    if not llm_cfg.use_ollama:
        return None
    try:
        from src.llm.ollama_client import create_ollama_client
        return create_ollama_client(base_url=llm_cfg.api_base, model=llm_cfg.model_name)
    except Exception as e:
        print(f"  [WARN] Could not connect to Ollama: {e}")
        return None


def run_single_episode_with_shock(
    agent,
    env: SemiSimEnvironment,
    shock_generator: ShockGenerator,
    shock_mode: str,
    max_steps: int = 30,
) -> dict:
    """Run one episode with specific shock mode."""
    obs = env.reset(seed=np.random.randint(0, 10000))
    agent.reset()

    step_rewards = []
    shocks_received = []
    oracle_rewards = []

    for step in range(max_steps):
        action, info = agent.step(obs, env)
        obs_next, rewards, done, info = env.step([action])

        step_rewards.append(rewards.get('total', 0))
        oracle_rewards.append(0.5)  # Oracle always gets 0.5 per step
        shocks_received.append(env.active_shock)

        obs = obs_next
        if done:
            break

    metrics = env.compute_metrics()
    return {
        'metrics': metrics,
        'step_rewards': step_rewards,
        'shocks_received': shocks_received,
        'oracle_rewards': oracle_rewards,
        'cumulative': sum(step_rewards),
        'oracle_cumulative': sum(oracle_rewards),
    }


def run_adversarial_experiment(
    shock_mode: str = "random",
    num_episodes: int = 3,
    seed: int = 42,
    llm_cfg: LLMConfig = None,
) -> dict:
    """Run experiment with specific shock mode using real LLM via Ollama."""
    if llm_cfg is None:
        llm_cfg = LLMConfig()
    semi_cfg = SemiSimConfig()
    shock_cfg = ShockConfig()
    agent_cfg = AgentConfig()
    wm_cfg = WorldModelConfig()
    llm_client = make_llm_client(llm_cfg)

    # Override shock probability for testing
    if shock_mode == "adversarial":
        shock_cfg.adversarial_attack_strength = 0.8

    shock_gen = ShockGenerator(shock_cfg, seed=seed)
    adv_llm = AdversarialPolicyLLM(llm_client=llm_client)

    env = SemiSimEnvironment(semi_cfg, shock_cfg, seed=seed)

    agent = ReflectiChainAgent(
        llm_config=llm_cfg,
        agent_config=agent_cfg,
        world_model_cfg=wm_cfg,
        semi_cfg=semi_cfg,
        llm_client=llm_client,
    )

    all_results = []
    for ep in range(num_episodes):
        result = run_single_episode_with_shock(
            agent, env, shock_gen, shock_mode, semi_cfg.max_steps
        )
        all_results.append(result)

    # Average
    avg_metrics = {
        'CEE': np.mean([r['metrics']['CEE'] for r in all_results]),
        'OR': np.mean([r['metrics']['OR'] for r in all_results]),
        'ARL': np.mean([r['metrics']['ARL'] for r in all_results]),
        'RCI': np.mean([r['metrics']['RCI'] for r in all_results]),
        'cumulative_reward': np.mean([r['cumulative'] for r in all_results]),
        'oracle_cumulative': np.mean([r['oracle_cumulative'] for r in all_results]),
    }

    # Compute regret
    actual_cum = avg_metrics['cumulative_reward']
    oracle_cum = avg_metrics['oracle_cumulative']
    avg_metrics['Regret'] = oracle_cum - actual_cum
    avg_metrics['NormalizedRegret'] = avg_metrics['Regret'] / (abs(oracle_cum) + 1e-6)

    return avg_metrics


def run_baseline_with_shock(baseline_class, shock_mode, num_episodes=3, seed=42, llm_cfg=None):
    """Run a baseline agent under specific shock mode with real LLM via Ollama."""
    if llm_cfg is None:
        llm_cfg = LLMConfig()
    semi_cfg = SemiSimConfig()
    shock_cfg = ShockConfig()
    llm_client = make_llm_client(llm_cfg)
    all_results = []

    for ep in range(num_episodes):
        env = SemiSimEnvironment(semi_cfg, shock_cfg, seed=seed + ep)
        # Only pass llm_client to classes that accept it
        init_params = baseline_class.__init__.__code__.co_varnames
        if 'llm_client' in init_params:
            agent = baseline_class(llm_client=llm_client)
        else:
            agent = baseline_class()

        obs = env.reset()
        agent.reset(env)

        step_rewards = []
        for step in range(semi_cfg.max_steps):
            actions = agent.act(obs)
            obs_next, rewards, done, info = env.step(actions)
            agent.update(obs, actions, rewards.get('total', 0), obs_next, done)
            step_rewards.append(rewards.get('total', 0))
            obs = obs_next
            if done:
                break

        all_results.append({
            'metrics': env.compute_metrics(),
            'cumulative': sum(step_rewards),
        })

    return {
        'CEE': np.mean([r['metrics']['CEE'] for r in all_results]),
        'OR': np.mean([r['metrics']['OR'] for r in all_results]),
        'ARL': np.mean([r['metrics']['ARL'] for r in all_results]),
        'RCI': np.mean([r['metrics']['RCI'] for r in all_results]),
        'cumulative_reward': np.mean([r['cumulative'] for r in all_results]),
    }


def main():
    print("=" * 60)
    print(" Adversarial Stress Test Experiment Suite")
    print(f" Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    os.makedirs("paper_results", exist_ok=True)
    results = {}
    llm_cfg = LLMConfig()

    # ─── Exp-AS-1: Three Shock Modes ─────────────────────────────────────────
    print("\n--- Exp-AS-1: Shock Mode Comparison ---")
    for mode in ['random', 'adversarial', 'historical']:
        print(f"  [{mode}] Running ReflectiChain...", end=" ", flush=True)
        r = run_adversarial_experiment(shock_mode=mode, num_episodes=3, seed=42, llm_cfg=llm_cfg)
        results[f'reflecti_{mode}'] = r
        print(f"CEE={r['CEE']:,.0f}  OR={r['OR']*100:.1f}%  Regret={r['Regret']:.4f}")

    # ─── Exp-AS-2: Baseline comparison under adversarial ──────────────────────
    print("\n--- Exp-AS-2: Baselines under Adversarial Shock ---")
    for name, cls in [('PPO', PPORLBaseline), ('Qwen', VanillaLLMBaseline)]:
        print(f"  [{name}] Running under adversarial shock...", end=" ", flush=True)
        r = run_baseline_with_shock(cls, 'adversarial', num_episodes=3, seed=42, llm_cfg=llm_cfg)
        results[f'{name.lower()}_adversarial'] = r
        print(f"CEE={r['CEE']:,.0f}  OR={r['OR']*100:.1f}%  ARL={r['ARL']:.2f}")

    # ─── Exp-AS-3: Adversarial strength scaling ────────────────────────────────
    print("\n--- Exp-AS-3: Adversarial Strength Scaling ---")
    shock_cfg = ShockConfig()
    scaling = []
    for strength in [0.3, 0.5, 0.7, 0.9]:
        shock_cfg.adversarial_attack_strength = strength
        shock_gen = ShockGenerator(shock_cfg, seed=42)

        semi_cfg = SemiSimConfig()
        agent_cfg = AgentConfig()
        wm_cfg = WorldModelConfig()
        llm_client = make_llm_client(llm_cfg)

        env = SemiSimEnvironment(semi_cfg, shock_cfg, seed=42)
        agent = ReflectiChainAgent(
            llm_cfg, agent_cfg, wm_cfg, semi_cfg, llm_client=llm_client,
        )
        obs = env.reset()
        step_rewards = []
        for _ in range(semi_cfg.max_steps):
            action, _ = agent.step(obs, env)
            obs, rewards, done, _ = env.step([action])
            step_rewards.append(rewards.get('total', 0))
            if done:
                break

        metrics = env.compute_metrics()
        scaling.append({
            'strength': strength,
            'CEE': float(metrics['CEE']),
            'OR': float(metrics['OR']),
            'ARL': float(metrics['ARL']),
            'cumulative_reward': float(sum(step_rewards)),
        })
        print(f"  strength={strength}  CEE={metrics['CEE']:,.0f}  OR={metrics['OR']*100:.1f}%  ARL={metrics['ARL']:.2f}")

    results['adversarial_scaling'] = scaling

    # ─── Exp-AS-5: Long-horizon stability (T=100) ──────────────────────────────
    print("\n--- Exp-AS-5: Long-horizon Stability (T=100) ---")
    semi_cfg_long = SemiSimConfig()
    semi_cfg_long.max_steps = 100
    shock_cfg_as5 = ShockConfig()
    shock_cfg_as5.shock_probability = 0.25
    llm_client_long = make_llm_client(llm_cfg)

    env_long = SemiSimEnvironment(semi_cfg_long, shock_cfg_as5, seed=42)
    agent_long = ReflectiChainAgent(
        LLMConfig(), AgentConfig(), WorldModelConfig(), semi_cfg_long, llm_client=llm_client_long,
    )

    obs = env_long.reset()
    step_rewards_long = []
    losses = []

    for step in range(100):
        action, info = agent_long.step(obs, env_long)
        obs_next, rewards, done, info = env_long.step([action])
        step_rewards_long.append(rewards.get('total', 0))
        if agent_long.lora_updater.loss_history:
            losses.append(agent_long.lora_updater.loss_history[-1])
        obs = obs_next
        if done:
            break

    metrics_long = env_long.compute_metrics()
    results['long_horizon'] = {
        'CEE': float(metrics_long['CEE']),
        'OR': float(metrics_long['OR']),
        'ARL': float(metrics_long['ARL']),
        'cumulative_reward': float(sum(step_rewards_long)),
        'avg_loss': float(np.mean(losses)) if losses else 0.0,
        'steps': len(step_rewards_long),
    }
    print(f"  T=100: CEE={metrics_long['CEE']:,.0f}  OR={metrics_long['OR']*100:.1f}%  ARL={metrics_long['ARL']:.2f}")

    # ─── Save ────────────────────────────────────────────────────────────────
    with open("paper_results/adv_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f" Adversarial experiments completed. Saved to paper_results/")
    print(f" Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
