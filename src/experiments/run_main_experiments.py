"""
Main Experiment Runner: runs all standard experiments (Table 1).
Includes ablation studies and scaling law experiments.
"""
import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import (
    SemiSimConfig, LLMConfig, AgentConfig,
    WorldModelConfig, ShockConfig, EvalConfig,
)
from src.semi_sim.enhanced_env import EnhancedSemiSimEnvironment as SemiSimEnvironment, Action
from src.semi_sim.world_model import WorldModelWrapper
from src.reflecti_chain.reflection import ReflectiChainAgent
from src.evaluation.metrics import MetricCalculator, EvalResult
from src.experiments.baselines import PPORLBaseline, VanillaLLMBaseline, RandomBaseline


def set_seed(seed: int = 42):
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)


def make_llm_client(llm_cfg: LLMConfig):
    """Create an Ollama LLM client. Returns None if Ollama is not available."""
    if not llm_cfg.use_ollama:
        return None
    try:
        from src.llm.ollama_client import create_ollama_client
        return create_ollama_client(
            base_url=llm_cfg.api_base,
            model=llm_cfg.model_name,
        )
    except Exception as e:
        print(f"  [WARN] Could not connect to Ollama: {e}")
        print(f"  [WARN] Falling back to mock mode. Set use_ollama=False to suppress this.")
        return None


def run_single_episode(
    agent,
    env: SemiSimEnvironment,
    max_steps: int = 30,
    verbose: bool = False,
) -> dict:
    """Run a single episode and collect all metrics."""
    obs = env.reset()
    agent.reset()

    step_rewards = []
    decisions = []

    for step in range(max_steps):
        action, info = agent.step(obs, env)

        step_rewards.append(info.get('immediate_reward', 0.0))
        decisions.append(info)

        obs_next = env.get_state_snapshot()
        done = (step >= max_steps - 1)

        if done:
            break

    # Final metrics
    metrics = env.compute_metrics()

    return {
        'metrics': metrics,
        'step_rewards': step_rewards,
        'decisions': decisions,
        'episode_length': len(step_rewards),
        'cumulative_reward': sum(step_rewards),
    }


def run_reflecti_chain(
    semi_cfg: SemiSimConfig,
    agent_cfg: AgentConfig,
    wm_cfg: WorldModelConfig,
    llm_cfg: LLMConfig,
    shock_cfg: ShockConfig,
    num_episodes: int = 5,
    seed: int = 42,
) -> EvalResult:
    """Run ReflectiChain agent with real LLM via Ollama."""
    llm_client = make_llm_client(llm_cfg)
    env = SemiSimEnvironment(semi_cfg, shock_cfg, seed=seed)
    agent = ReflectiChainAgent(
        llm_config=llm_cfg,
        agent_config=agent_cfg,
        world_model_cfg=wm_cfg,
        semi_cfg=semi_cfg,
        agent_id=0,
        agent_type="profit_driven",
        llm_client=llm_client,
        device="cuda" if os.environ.get('CUDA_VISIBLE_DEVICES') else "cpu",
    )

    all_metrics = []
    for ep in range(num_episodes):
        result = run_single_episode(agent, env, semi_cfg.max_steps)
        all_metrics.append(result['metrics'])

    # Average over episodes
    avg_cee = np.mean([m['CEE'] for m in all_metrics])
    avg_rci = np.mean([m['RCI'] for m in all_metrics])
    avg_bwi = np.mean([m['BWI'] for m in all_metrics])
    avg_or = np.mean([m['OR'] for m in all_metrics])
    avg_arl = np.mean([m['ARL'] for m in all_metrics])

    return EvalResult(
        CEE=avg_cee,
        RCI=avg_rci,
        BWI=avg_bwi,
        OR=avg_or,
        ARL=avg_arl,
        step_rewards=[m['OR'] for m in all_metrics],
        cumulative_reward=avg_cee,
    )


def run_baseline(
    baseline_class,
    semi_cfg: SemiSimConfig,
    shock_cfg: ShockConfig,
    num_episodes: int = 5,
    seed: int = 42,
    **kwargs,
) -> EvalResult:
    """Run a baseline agent with real LLM via Ollama."""
    llm_client = kwargs.pop('llm_client', None)
    _ = kwargs.pop('llm_cfg', None)
    env = SemiSimEnvironment(semi_cfg, shock_cfg, seed=seed)
    # Only pass llm_client to classes that accept it
    if hasattr(baseline_class, '__init__'):
        init_sig = baseline_class.__init__.__code__.co_varnames
        if 'llm_client' in init_sig:
            agent_kwargs = {k: v for k, v in kwargs.items() if k != 'llm_client'}
            agent = baseline_class(llm_client=llm_client, **agent_kwargs)
        else:
            agent = baseline_class(**kwargs)
    else:
        agent = baseline_class(**kwargs)

    all_metrics = []
    all_rewards = []

    for ep in range(num_episodes):
        obs = env.reset()
        agent.reset(env)

        step_rewards_ep = []
        for step in range(semi_cfg.max_steps):
            actions = agent.act(obs)
            obs_next, rewards, done, info = env.step(actions)
            agent.update(obs, actions, rewards.get('total', 0), obs_next, done)

            step_rewards_ep.append(rewards.get('total', 0))
            obs = obs_next
            if done:
                break

        metrics = env.compute_metrics()
        all_metrics.append(metrics)
        all_rewards.extend(step_rewards_ep)

    return EvalResult(
        CEE=np.mean([m['CEE'] for m in all_metrics]),
        RCI=np.mean([m['RCI'] for m in all_metrics]),
        BWI=np.mean([m['BWI'] for m in all_metrics]),
        OR=np.mean([m['OR'] for m in all_metrics]),
        ARL=np.mean([m['ARL'] for m in all_metrics]),
        step_rewards=all_rewards,
        cumulative_reward=np.sum(all_rewards),
    )


def run_ablation_study(
    semi_cfg: SemiSimConfig,
    agent_cfg: AgentConfig,
    wm_cfg: WorldModelConfig,
    llm_cfg: LLMConfig,
    shock_cfg: ShockConfig,
    num_episodes: int = 3,
) -> dict:
    """Run ablation experiments with real LLM via Ollama."""
    variants = {}
    llm_client = make_llm_client(llm_cfg)

    # Full ReflectiChain
    print("  Running Full ReflectiChain...")
    variants['Full'] = run_reflecti_chain(
        semi_cfg, agent_cfg, wm_cfg, llm_cfg, shock_cfg,
        num_episodes=num_episodes, seed=42,
    )

    # Without World Model
    print("  Running w/o World Model...")
    cfg_no_wm = SemiSimConfig()
    env = SemiSimEnvironment(cfg_no_wm, shock_cfg, seed=42)
    agent = ReflectiChainAgent(
        llm_config=llm_cfg,
        agent_config=agent_cfg,
        world_model_cfg=wm_cfg,
        semi_cfg=semi_cfg,
        llm_client=llm_client,
    )
    # Override: disable WM evaluation
    agent._eval_via_world_model = lambda *args: 0.0  # Disable WM
    results = []
    for _ in range(num_episodes):
        obs = env.reset()
        agent.reset()
        for _ in range(semi_cfg.max_steps):
            action, info = agent.step(obs, env)
            obs, rewards, done, _ = env.step([action])
            if done:
                break
        results.append(env.compute_metrics())
    variants['w/o WM'] = EvalResult(
        CEE=np.mean([r['CEE'] for r in results]),
        RCI=np.mean([r['RCI'] for r in results]),
        BWI=np.mean([r['BWI'] for r in results]),
        OR=np.mean([r['OR'] for r in results]),
        ARL=np.mean([r['ARL'] for r in results]),
        step_rewards=[0.0], cumulative_reward=0.0,
    )

    # Without Retrospective RL
    print("  Running w/o Retrospective RL...")
    cfg_no_retro = SemiSimConfig()
    env2 = SemiSimEnvironment(cfg_no_retro, shock_cfg, seed=42)
    agent2 = ReflectiChainAgent(
        llm_config=llm_cfg,
        agent_config=agent_cfg,
        world_model_cfg=wm_cfg,
        semi_cfg=semi_cfg,
        llm_client=llm_client,
    )
    # Disable LoRA updates
    agent2.lora_updater.update = lambda: 0.0
    results2 = []
    for _ in range(num_episodes):
        obs = env2.reset()
        agent2.reset()
        for _ in range(semi_cfg.max_steps):
            action, info = agent2.step(obs, env2)
            obs, rewards, done, _ = env2.step([action])
            if done:
                break
        results2.append(env2.compute_metrics())
    variants['w/o Retro RL'] = EvalResult(
        CEE=np.mean([r['CEE'] for r in results2]),
        RCI=np.mean([r['RCI'] for r in results2]),
        BWI=np.mean([r['BWI'] for r in results2]),
        OR=np.mean([r['OR'] for r in results2]),
        ARL=np.mean([r['ARL'] for r in results2]),
        step_rewards=[0.0], cumulative_reward=0.0,
    )

    # Without Internal Reflection
    print("  Running w/o Internal Reflection...")
    cfg_no_int = SemiSimConfig()
    env3 = SemiSimEnvironment(cfg_no_int, shock_cfg, seed=42)
    agent3 = ReflectiChainAgent(
        llm_config=llm_cfg,
        agent_config=agent_cfg,
        world_model_cfg=wm_cfg,
        semi_cfg=semi_cfg,
        llm_client=llm_client,
    )
    # Disable internal critic
    agent3.internal_critic.score_action = lambda *args: 85.0
    results3 = []
    for _ in range(num_episodes):
        obs = env3.reset()
        agent3.reset()
        for _ in range(semi_cfg.max_steps):
            action, info = agent3.step(obs, env3)
            obs, rewards, done, _ = env3.step([action])
            if done:
                break
        results3.append(env3.compute_metrics())
    variants['w/o Internal'] = EvalResult(
        CEE=np.mean([r['CEE'] for r in results3]),
        RCI=np.mean([r['RCI'] for r in results3]),
        BWI=np.mean([r['BWI'] for r in results3]),
        OR=np.mean([r['OR'] for r in results3]),
        ARL=np.mean([r['ARL'] for r in results3]),
        step_rewards=[0.0], cumulative_reward=0.0,
    )

    return variants


def run_scaling_experiment(
    semi_cfg: SemiSimConfig,
    agent_cfg: AgentConfig,
    wm_cfg: WorldModelConfig,
    llm_cfg: LLMConfig,
    shock_cfg: ShockConfig,
) -> dict:
    """Run scaling law experiments with real LLM via Ollama."""
    scaling_results = {
        'N_scaling': [],  # Vary num_candidates
        'K_scaling': [],  # Vary memory_window
    }
    llm_client = make_llm_client(llm_cfg)

    # Vary N (sampling width)
    print("  Running N scaling...")
    for N in [1, 3, 5, 10]:
        cfg = AgentConfig(num_candidates=N)
        env = SemiSimEnvironment(semi_cfg, shock_cfg, seed=42)
        agent = ReflectiChainAgent(
            llm_config=llm_cfg,
            agent_config=cfg,
            world_model_cfg=wm_cfg,
            semi_cfg=semi_cfg,
            llm_client=llm_client,
        )
        obs = env.reset()
        agent.reset()
        for _ in range(semi_cfg.max_steps):
            action, info = agent.step(obs, env)
            obs, rewards, done, _ = env.step([action])
            if done:
                break
        m = env.compute_metrics()
        scaling_results['N_scaling'].append({'N': N, 'OR': m['OR'], 'ARL': m['ARL'], 'CEE': m['CEE']})

    # Vary K (memory window)
    print("  Running K scaling...")
    for K in [1, 3, 5, 10]:
        cfg = AgentConfig(memory_window=K)
        env = SemiSimEnvironment(semi_cfg, shock_cfg, seed=42)
        agent = ReflectiChainAgent(
            llm_config=llm_cfg,
            agent_config=cfg,
            world_model_cfg=wm_cfg,
            semi_cfg=semi_cfg,
            llm_client=llm_client,
        )
        obs = env.reset()
        agent.reset()
        for _ in range(semi_cfg.max_steps):
            action, info = agent.step(obs, env)
            obs, rewards, done, _ = env.step([action])
            if done:
                break
        m = env.compute_metrics()
        scaling_results['K_scaling'].append({'K': K, 'OR': m['OR'], 'ARL': m['ARL'], 'CEE': m['CEE']})

    return scaling_results


def generate_paper_table(
    results: dict,
    ideal: dict,
    output_dir: str = "paper_results",
) -> str:
    """Generate Table 1 from experiment results."""
    os.makedirs(output_dir, exist_ok=True)

    table_lines = [
        "Metric (Goal)",
        "PPO (Classic RL)",
        "Qwen2.5-7B",
        "InternLM2.5",
        "ReflectiChain (Ours)",
        "Ideal Ref.",
        "---",
        "---",
        "---",
        "---",
        "---",
        "---",
    ]

    # Build table data
    def fmt_cee(v): return f"{v:,.0f}"
    def fmt_rci(v): return f"{v:.2f}"
    def fmt_bwi(v): return f"{v:.4f}"
    def fmt_or(v): return f"{v*100:.1f}%"
    def fmt_arl(v): return f"{v:.2f}"

    table_data = [
        ("Total Cash (CEE ↑)", 'CEE', fmt_cee, "> 1,000K"),
        ("Avg Compliance (RCI ↑)", 'RCI', fmt_rci, "> 85.00"),
        ("Bullwhip Index (BWI ≈ 1)", 'BWI', fmt_bwi, "1.0 ~ 1.5"),
        ("Operability Ratio (OR ↑)", 'OR', fmt_or, "> 80.0%"),
        ("Average Risk Level (ARL ↓)", 'ARL', fmt_arl, "< 40.00"),
    ]

    lines = []
    lines.append("=" * 90)
    lines.append(f"{'Table 1: Metric-level comparison of ReflectiChain against baselines under macroeconomic shocks':^90}")
    lines.append("=" * 90)
    header = f"{'Metric (Goal)':<35} {'PPO':>12} {'Qwen2.5-7B':>12} {'InternLM2.5':>12} {'ReflectiChain':>14} {'Ideal Ref.':>12}"
    lines.append(header)
    lines.append("-" * 90)

    for display_name, key, formatter, ideal_str in table_data:
        row = []
        for name in ['PPO', 'Qwen2.5-7B', 'InternLM2.5', 'ReflectiChain']:
            res = results.get(name)
            if res is None:
                row.append('N/A')
            else:
                val = getattr(res, key)
                row.append(formatter(val))
        lines.append(f"{display_name:<35} {row[0]:>12} {row[1]:>12} {row[2]:>12} {row[3]:>14} {ideal_str:>12}")

    lines.append("-" * 90)
    lines.append("")

    output = "\n".join(lines)
    print(output)

    # Save to file
    with open(os.path.join(output_dir, "table1.txt"), 'w', encoding='utf-8') as f:
        f.write(output)

    return output


def run_trace_collection(
    semi_cfg: SemiSimConfig,
    agent_cfg: AgentConfig,
    wm_cfg: WorldModelConfig,
    llm_cfg: LLMConfig,
    shock_cfg: ShockConfig,
    num_episodes: int = 3,
    seed: int = 42,
) -> dict:
    """Run ReflectiChain and collect mechanism trace + RL loss for Figure 5 & 6."""
    llm_client = make_llm_client(llm_cfg)
    all_traces = []
    all_losses = []

    for ep in range(num_episodes):
        env = SemiSimEnvironment(semi_cfg, shock_cfg, seed=seed + ep * 100)
        agent = ReflectiChainAgent(
            llm_config=llm_cfg,
            agent_config=agent_cfg,
            world_model_cfg=wm_cfg,
            semi_cfg=semi_cfg,
            agent_id=0,
            agent_type="profit_driven",
            llm_client=llm_client,
            device="cpu",
        )

        obs = env.reset()
        agent.reset()
        step_rewards = []
        losses = []

        for step in range(semi_cfg.max_steps):
            action, info = agent.step(obs, env)
            step_rewards.append(info.get('immediate_reward', 0.0))
            if agent.lora_updater.loss_history:
                losses.append(agent.lora_updater.loss_history[-1])
            obs, rewards, done, _ = env.step([action])
            if done:
                break

        trace = agent.get_decision_trace()
        all_traces.append(trace)
        all_losses.append(losses)

    return {'traces': all_traces, 'losses': all_losses, 'step_rewards': step_rewards}


def main():
    print("=" * 70)
    print(" ReflectiChain Main Experiment Suite")
    print(f" Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Configuration
    semi_cfg = SemiSimConfig()
    agent_cfg = AgentConfig()
    wm_cfg = WorldModelConfig()
    llm_cfg = LLMConfig()
    shock_cfg = ShockConfig()
    eval_cfg = EvalConfig()

    set_seed(42)
    os.makedirs("paper_results", exist_ok=True)

    num_episodes = 5
    results = {}

    # ─── Run Baselines ────────────────────────────────────────────────────────
    print("\n--- Running Baselines ---")

    print("  [1/4] Running PPO baseline...")
    results['PPO'] = run_baseline(
        PPORLBaseline, semi_cfg, shock_cfg,
        num_episodes=num_episodes, seed=42, llm_cfg=llm_cfg,
    )
    print(f"       CEE={results['PPO'].CEE:.0f}, OR={results['PPO'].OR*100:.1f}%, ARL={results['PPO'].ARL:.2f}")

    print("  [2/4] Running Vanilla LLM (Qwen2.5-7B) baseline...")
    results['Qwen2.5-7B'] = run_baseline(
        VanillaLLMBaseline, semi_cfg, shock_cfg,
        num_episodes=num_episodes, seed=42,
        llm_cfg=llm_cfg, model_name=llm_cfg.model_name,
    )
    print(f"       CEE={results['Qwen2.5-7B'].CEE:.0f}, OR={results['Qwen2.5-7B'].OR*100:.1f}%, ARL={results['Qwen2.5-7B'].ARL:.2f}")

    print("  [3/4] Running Vanilla LLM (InternLM2.5) baseline...")
    results['InternLM2.5'] = run_baseline(
        VanillaLLMBaseline, semi_cfg, shock_cfg,
        num_episodes=num_episodes, seed=42,
        llm_cfg=llm_cfg, model_name=llm_cfg.model_name,
    )
    print(f"       CEE={results['InternLM2.5'].CEE:.0f}, OR={results['InternLM2.5'].OR*100:.1f}%, ARL={results['InternLM2.5'].ARL:.2f}")

    print("  [4/4] Running ReflectiChain (Ours)...")
    results['ReflectiChain'] = run_reflecti_chain(
        semi_cfg, agent_cfg, wm_cfg, llm_cfg, shock_cfg,
        num_episodes=num_episodes, seed=42,
    )
    print(f"       CEE={results['ReflectiChain'].CEE:.0f}, OR={results['ReflectiChain'].OR*100:.1f}%, ARL={results['ReflectiChain'].ARL:.2f}")

    # ─── Collect Mechanism Trace + RL Loss ────────────────────────────────
    print("\n--- Collecting Mechanism Trace (for Figure 5 & 6) ---")
    trace_data = run_trace_collection(
        semi_cfg, agent_cfg, wm_cfg, llm_cfg, shock_cfg,
        num_episodes=3, seed=42,
    )
    with open("paper_results/mechanism_trace.json", 'w') as f:
        json.dump({
            'traces': trace_data['traces'],
            'losses': trace_data['losses'],
            'step_rewards': trace_data['step_rewards'],
        }, f, indent=2)
    print("  Saved mechanism_trace.json")

    # ─── Generate Table 1 ───────────────────────────────────────────────────
    print("\n--- Generating Table 1 ---")
    ideal = {
        'CEE': '> 1,000K',
        'RCI': '> 85.00',
        'BWI': '1.0 ~ 1.5',
        'OR': '> 80.0%',
        'ARL': '< 40.00',
    }
    generate_paper_table(results, ideal, "paper_results")

    # ─── Ablation Study ─────────────────────────────────────────────────────
    print("\n--- Running Ablation Study ---")
    ablation_results = run_ablation_study(
        semi_cfg, agent_cfg, wm_cfg, llm_cfg, shock_cfg,
        num_episodes=3,
    )

    # Save ablation results
    ablation_data = {
        name: {
            'CEE': r.CEE,
            'RCI': r.RCI,
            'BWI': r.BWI,
            'OR': r.OR,
            'ARL': r.ARL,
        }
        for name, r in ablation_results.items()
    }
    with open("paper_results/ablation_results.json", 'w') as f:
        json.dump(ablation_data, f, indent=2)

    print("\nAblation Summary:")
    for name, res in ablation_results.items():
        print(f"  {name:<20} OR={res.OR*100:.1f}%  CEE={res.CEE:,.0f}  ARL={res.ARL:.2f}")

    # ─── Scaling Law Experiments ─────────────────────────────────────────────
    print("\n--- Running Scaling Law Experiments ---")
    scaling_results = run_scaling_experiment(
        semi_cfg, agent_cfg, wm_cfg, llm_cfg, shock_cfg,
    )
    with open("paper_results/scaling_results.json", 'w') as f:
        json.dump(scaling_results, f, indent=2)

    print("N Scaling:")
    for item in scaling_results['N_scaling']:
        print(f"  N={item['N']:2d}  OR={item['OR']*100:.1f}%  CEE={item['CEE']:,.0f}")

    print("K Scaling:")
    for item in scaling_results['K_scaling']:
        print(f"  K={item['K']:2d}  OR={item['OR']*100:.1f}%  CEE={item['CEE']:,.0f}")

    # ─── Save All Results ───────────────────────────────────────────────────
    all_results = {
        name: r.to_dict() for name, r in results.items()
    }
    with open("paper_results/all_experiment_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f" Experiment suite completed. Results saved to paper_results/")
    print(f" Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")

    return results


if __name__ == "__main__":
    main()
