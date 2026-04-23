"""
Multi-Agent Game Experiment Suite.
Tests ReflectiChain under competitive/cooperative multi-agent scenarios.
"""
import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import (
    SemiSimConfig, LLMConfig, AgentConfig,
    WorldModelConfig, ShockConfig, MultiAgentConfig,
)
from src.semi_sim.enhanced_env import EnhancedSemiSimEnvironment as SemiSimEnvironment, Action
from src.semi_sim.multi_agent import MultiAgentInterface
from src.reflecti_chain.reflection import ReflectiChainAgent
from src.evaluation.metrics import MetricCalculator


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


def run_single_ma_episode(
    agents: list,
    ma_interface: MultiAgentInterface,
    env: SemiSimEnvironment,
    max_steps: int = 30,
) -> dict:
    """Run one episode with multiple ReflectiChain agents."""
    obs = env.reset()
    for agent in agents:
        agent.reset()

    step_rewards = []
    agent_cash_history = {aid: [] for aid in range(len(agents))}

    for step in range(max_steps):
        # Broadcast state
        broadcasts = ma_interface.broadcast_state(obs)

        # Each agent selects action
        agent_actions = {}
        for aid, agent in enumerate(agents):
            action, _ = agent.step(obs, env)
            agent_actions[aid] = [action]

        # Resolve conflicts
        merged_actions, conflict_log = ma_interface.collect_actions(agent_actions)

        # Execute
        obs_next, rewards, done, info = env.step(merged_actions)

        # Track per-agent cash
        for aid in range(len(agents)):
            agent_cash_history[aid].append(
                env.node_states[aid].cash if aid in env.node_states else 0.0
            )

        step_rewards.append(rewards)
        obs = obs_next
        if done:
            break

    # Build episode history for game metrics
    episode_history = []
    for t in range(len(step_rewards)):
        episode_history.append({
            'timestep': t,
            'agent_cash': {aid: agent_cash_history[aid][t] if t < len(agent_cash_history[aid]) else 0
                           for aid in agent_cash_history},
        })

    game_metrics = ma_interface.compute_game_metrics(
        episode_history, list(range(len(agents)))
    )
    env_metrics = env.compute_metrics()

    return {
        'game_metrics': game_metrics,
        'env_metrics': env_metrics,
        'step_rewards': step_rewards,
        'episode_history': episode_history,
        'conflict_log': conflict_log,
    }


def run_ma_experiment(
    num_agents: int = 3,
    game_type: str = "zero_sum",
    num_episodes: int = 3,
    seed: int = 42,
    llm_cfg: LLMConfig = None,
) -> dict:
    if llm_cfg is None:
        llm_cfg = LLMConfig()
    llm_client = make_llm_client(llm_cfg)
    semi_cfg = SemiSimConfig()
    shock_cfg = ShockConfig()
    agent_cfg = AgentConfig()
    wm_cfg = WorldModelConfig()

    agent_types = ['profit_driven', 'resilience_driven', 'compliance_driven'][:num_agents]
    all_game = []
    all_env = []

    for ep in range(num_episodes):
        ep_seed = seed + ep
        env = SemiSimEnvironment(semi_cfg, shock_cfg, seed=ep_seed)

        # Build agents
        agents = []
        for aid in range(num_agents):
            agent = ReflectiChainAgent(
                llm_config=llm_cfg,
                agent_config=agent_cfg,
                world_model_cfg=wm_cfg,
                semi_cfg=semi_cfg,
                agent_id=aid,
                agent_type=agent_types[aid % len(agent_types)],
                llm_client=llm_client,
            )
            agents.append(agent)

        # Interface
        ma_config = MultiAgentConfig(
            num_agents=num_agents,
            enable_cooperation=(game_type != "zero_sum"),
        )
        ma_interface = MultiAgentInterface(ma_config, semi_cfg, env)

        result = run_single_ma_episode(agents, ma_interface, env, semi_cfg.max_steps)
        all_game.append(result['game_metrics'])
        all_env.append(result['env_metrics'])

    avg_game = {
        'SW': float(np.mean([m['SW'] for m in all_game])),
        'CI': float(np.mean([m['CI'] for m in all_game])),
        'epsilon_NE': float(np.mean([m['epsilon_NE'] for m in all_game])),
        'CR': float(np.mean([m['CR'] for m in all_game])),
    }
    avg_env = {
        'CEE': float(np.mean([m['CEE'] for m in all_env])),
        'OR': float(np.mean([m['OR'] for m in all_env])),
        'ARL': float(np.mean([m['ARL'] for m in all_env])),
        'RCI': float(np.mean([m['RCI'] for m in all_env])),
    }

    return {'game_metrics': avg_game, 'env_metrics': avg_env}


def main():
    print("=" * 60)
    print(" Multi-Agent Game Experiment Suite")
    print(f" Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    os.makedirs("paper_results", exist_ok=True)
    llm_cfg = LLMConfig()
    results = {}

    # ─── Exp-MA-1: Multi-Agent baseline comparison ───────────────────────────
    print("\n--- Exp-MA-1: 2/3/5-Agent Comparison ---")
    for num_agents in [2, 3, 5]:
        print(f"  [{num_agents}-Agent] Running mixed game...", end=" ", flush=True)
        r = run_ma_experiment(num_agents=num_agents, game_type="mixed", num_episodes=3, seed=42, llm_cfg=llm_cfg)
        results[f'{num_agents}_agents'] = r
        print(f"SW={r['game_metrics']['SW']:,.0f}  CI={r['game_metrics']['CI']:.4f}  e-NE={r['game_metrics']['epsilon_NE']:.4f}")

    # ─── Exp-MA-2: Heterogeneous agent types ────────────────────────────────
    print("\n--- Exp-MA-2: Game Type Comparison ---")
    for game_type in ['zero_sum', 'cooperative', 'mixed']:
        print(f"  [{game_type}] Running 3-agent game...", end=" ", flush=True)
        r = run_ma_experiment(num_agents=3, game_type=game_type, num_episodes=3, seed=42, llm_cfg=llm_cfg)
        results[f'game_{game_type}'] = r
        print(f"SW={r['game_metrics']['SW']:,.0f}  CR={r['game_metrics']['CR']:.4f}")

    # ─── Exp-MA-4: Double-Loop ablation ──────────────────────────────────────
    print("\n--- Exp-MA-4: Double-Loop Ablation in Multi-Agent ---")
    semi_cfg = SemiSimConfig()
    shock_cfg = ShockConfig()
    agent_cfg = AgentConfig()
    wm_cfg = WorldModelConfig()
    llm_client = make_llm_client(llm_cfg)

    # Full
    print("  [Full Double-Loop] Running...", end=" ", flush=True)
    r_full = run_ma_experiment(num_agents=3, game_type="mixed", num_episodes=3, seed=42, llm_cfg=llm_cfg)
    print(f"SW={r_full['game_metrics']['SW']:,.0f}  e-NE={r_full['game_metrics']['epsilon_NE']:.4f}")

    # No LoRA
    print("  [No LoRA] Running...", end=" ", flush=True)
    env = SemiSimEnvironment(semi_cfg, shock_cfg, seed=42)
    agents_no_lora = []
    for aid in range(3):
        agent = ReflectiChainAgent(
            llm_config=llm_cfg,
            agent_config=agent_cfg,
            world_model_cfg=wm_cfg,
            semi_cfg=semi_cfg,
            agent_id=aid,
            agent_type=['profit_driven', 'resilience_driven', 'compliance_driven'][aid],
            llm_client=llm_client,
        )
        agent.lora_updater.update = lambda: 0.0
        agents_no_lora.append(agent)

    ma_config = MultiAgentConfig(num_agents=3, enable_cooperation=True)
    ma_interface = MultiAgentInterface(ma_config, semi_cfg, env)
    r_no_lora = run_single_ma_episode(agents_no_lora, ma_interface, env)
    print(f"SW={r_no_lora['game_metrics']['SW']:,.0f}  e-NE={r_no_lora['game_metrics']['epsilon_NE']:.4f}")

    results['ablation_double_loop'] = {
        'full': r_full['game_metrics'],
        'no_lora': r_no_lora['game_metrics'],
    }

    # ─── Exp-MA-5: Competition intensity scaling ──────────────────────────────
    print("\n--- Exp-MA-5: Competition Intensity Scaling ---")
    scaling = []
    for intensity in [0.2, 0.5, 0.8, 1.0]:
        ma_config_s = MultiAgentConfig(num_agents=3, competition_intensity=intensity, enable_cooperation=False)
        env_s = SemiSimEnvironment(semi_cfg, shock_cfg, seed=42)
        ma_int = MultiAgentInterface(ma_config_s, semi_cfg, env_s)
        agents_s = [
            ReflectiChainAgent(llm_cfg, agent_cfg, wm_cfg, semi_cfg, agent_id=i,
                agent_type=['profit_driven', 'resilience_driven', 'compliance_driven'][i], llm_client=llm_client)
            for i in range(3)
        ]
        r_s = run_single_ma_episode(agents_s, ma_int, env_s)
        scaling.append({
            'intensity': intensity,
            'SW': float(r_s['game_metrics']['SW']),
            'CI': float(r_s['game_metrics']['CI']),
        })
        print(f"  intensity={intensity}  SW={r_s['game_metrics']['SW']:,.0f}  CI={r_s['game_metrics']['CI']:.4f}")
    results['scaling_competition'] = scaling

    with open("paper_results/ma_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f" Multi-Agent experiments completed.")
    print(f" Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
