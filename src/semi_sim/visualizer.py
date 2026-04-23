"""
Trajectory Visualizer: generates supply chain risk propagation visualizations.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import networkx as nx
from typing import List, Dict, Optional, Tuple, Any
import os
from pathlib import Path


class TrajectoryVisualizer:
    """
    Generates visualizations for supply chain risk propagation trajectories.
    Supports:
        - Network topology with risk heatmap
        - Time-series metric plots
        - Agent performance comparison
        - Shock event timeline
    """

    def __init__(self, output_dir: str = "paper_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = {
            'upstream': '#E74C3C',
            'midstream': '#F39C12',
            'downstream': '#27AE60',
            'shock': '#8E44AD',
            'normal': '#3498DB',
        }

    def plot_network_risk(
        self,
        G: nx.DiGraph,
        node_states: Dict[int, Any],
        timestep: int,
        save_path: Optional[str] = None,
        show_labels: bool = True,
    ) -> str:
        """
        Plot supply chain network with risk heatmap coloring.

        Returns:
            path to saved figure
        """
        fig, ax = plt.subplots(figsize=(10, 7))

        # Create risk-based color map
        node_colors = []
        node_sizes = []
        labels = {}

        for nid in G.nodes():
            state = node_states.get(nid)
            if state is not None:
                # Risk: red=high, blue=low
                risk = getattr(state, 'risk_exposure', 0.5)
                color = plt.cm.RdYlBu_r(risk)
                node_colors.append(color)
                node_sizes.append(2000 + risk * 3000)
                labels[nid] = getattr(state, 'name', f'Node {nid}')
            else:
                node_colors.append('#CCCCCC')
                node_sizes.append(1000)
                labels[nid] = f'Node {nid}'

        # Layout
        pos = self._get_supply_chain_layout(G)

        # Draw edges with flow width
        edge_weights = []
        for u, v in G.edges():
            flow = G.edges[u, v].get('current_flow', 0)
            base_flow = G.edges[u, v].get('base_flow', 1)
            ratio = min(flow / (base_flow + 1e-6), 2.0)
            edge_weights.append(max(0.5, ratio * 2.0))

        # Normalize edge weights for visualization
        max_w = max(edge_weights) if edge_weights else 1
        edge_widths = [w / max_w * 4 for w in edge_weights]

        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color='#666666',
            width=edge_widths,
            alpha=0.6,
            arrows=True,
            arrowsize=15,
            connectionstyle='arc3,rad=0.1',
        )

        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors='white',
            linewidths=2,
        )

        if show_labels:
            nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=10, font_weight='bold')

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
        cbar.set_label('Risk Exposure', fontsize=11)

        ax.set_title(f'Supply Chain Risk Map — Step {timestep}', fontsize=14, fontweight='bold')
        ax.axis('off')

        path = save_path or str(self.output_dir / f'risk_map_t{timestep:03d}.png')
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path

    def _get_supply_chain_layout(self, G: nx.DiGraph) -> Dict[int, Tuple[float, float]]:
        """Custom layout for supply chain: Upstream -> Midstream -> Downstream."""
        pos = {}
        # Group nodes by type
        types = nx.get_node_attributes(G, 'type')
        upstream = [n for n in G.nodes() if types.get(n) == 'Upstream']
        midstream = [n for n in G.nodes() if types.get(n) == 'Midstream']
        downstream = [n for n in G.nodes() if types.get(n) == 'Downstream']

        for i, nid in enumerate(upstream):
            pos[nid] = (1.0, 2.0 - i * 1.2)
        for i, nid in enumerate(midstream):
            pos[nid] = (3.0, 1.5 - i * 1.0)
        for i, nid in enumerate(downstream):
            pos[nid] = (5.0, 1.5 - i * 1.2)

        return pos

    def plot_metrics_timeseries(
        self,
        timesteps: List[int],
        metrics: Dict[str, List[float]],
        metric_names: Dict[str, str],
        title: str = "Simulation Metrics Over Time",
        save_path: Optional[str] = None,
    ) -> str:
        """
        Plot multiple metrics over simulation timesteps.

        Args:
            timesteps: List of timestep indices
            metrics: Dict[metric_name -> List of values]
            metric_names: Dict[metric_name -> display name]
        """
        num_metrics = len(metrics)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 4 * num_metrics), sharex=True)

        if num_metrics == 1:
            axes = [axes]

        line_styles = ['-', '--', '-.', ':', '-']
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']

        for i, (mname, values) in enumerate(metrics.items()):
            ax = axes[i]
            ax.plot(timesteps, values, linestyle=line_styles[i % len(line_styles)],
                    color=colors[i % len(colors)], linewidth=2, label=metric_names.get(mname, mname))
            ax.fill_between(timesteps, values, alpha=0.15, color=colors[i % len(colors)])
            ax.set_ylabel(metric_names.get(mname, mname), fontsize=11)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Timestep', fontsize=11)
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()

        path = save_path or str(self.output_dir / 'metrics_timeseries.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path

    def plot_game_metrics_comparison(
        self,
        agent_names: List[str],
        game_metrics: Dict[str, float],
        save_path: Optional[str] = None,
    ) -> str:
        """Plot game-theoretic metrics comparison."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Social Welfare bar
        ax = axes[0]
        sw_data = [game_metrics.get(f'SW_{an}', 0) for an in agent_names]
        colors = ['#3498DB', '#E74C3C', '#27AE60', '#F39C12', '#8E44AD'][:len(agent_names)]
        ax.bar(agent_names, sw_data, color=colors, alpha=0.8, edgecolor='white')
        ax.set_ylabel('Social Welfare (Cumulative Cash)')
        ax.set_title('Social Welfare by Agent')
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, axis='y', alpha=0.3)

        # Nash Equilibrium gap
        ax = axes[1]
        ne_data = [game_metrics.get(f'epsilon_NE_{an}', 0) for an in agent_names]
        ax.bar(agent_names, ne_data, color=colors, alpha=0.8, edgecolor='white')
        ax.set_ylabel('ε-NE (Nash Gap)')
        ax.set_title('Nash Equilibrium Approximation')
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, axis='y', alpha=0.3)

        # Competition Intensity
        ax = axes[2]
        ci_data = [game_metrics.get(f'CI_{an}', 0) for an in agent_names]
        ax.bar(agent_names, ci_data, color=colors, alpha=0.8, edgecolor='white')
        ax.set_ylabel('Competition Intensity')
        ax.set_title('Strategy Differentiation')
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, axis='y', alpha=0.3)

        fig.suptitle('Multi-Agent Game-Theoretic Evaluation', fontsize=14, fontweight='bold')
        plt.tight_layout()

        path = save_path or str(self.output_dir / 'game_metrics.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path

    def plot_ablation_results(
        self,
        variants: List[str],
        step_rewards: List[float],
        retro_scores: List[float],
        failure_modes: List[str],
        save_path: Optional[str] = None,
    ) -> str:
        """Plot ablation study results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        x = np.arange(len(variants))
        width = 0.35

        ax = axes[0]
        bars1 = ax.bar(x - width/2, step_rewards, width, label='Step Reward', color='#3498DB', alpha=0.8)
        ax.set_ylabel('Step Reward')
        ax.set_title('Step Reward by Ablation Variant')
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=30, ha='right')
        ax.grid(True, axis='y', alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        # Annotate failure modes
        for i, (bar, fm) in enumerate(zip(bars1, failure_modes)):
            ax.annotate(fm, xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7, color='#666666')

        ax = axes[1]
        bars2 = ax.bar(x + width/2, retro_scores, width, label='Retro Score', color='#E74C3C', alpha=0.8)
        ax.set_ylabel('Retrospective Score')
        ax.set_title('Retrospective Score by Ablation Variant')
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=30, ha='right')
        ax.grid(True, axis='y', alpha=0.3)

        fig.suptitle('Ablation Study: Component Contribution Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()

        path = save_path or str(self.output_dir / 'ablation_results.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path

    def plot_scaling_curves(
        self,
        param_values: List[float],
        metrics_by_param: Dict[str, List[float]],
        param_name: str,
        metric_name: str,
        save_path: Optional[str] = None,
    ) -> str:
        """Plot scaling law curves (N vs performance, K vs convergence, etc.)."""
        fig, ax = plt.subplots(figsize=(9, 5))

        colors = ['#2E86AB', '#A23B72', '#F18F01']
        for i, (series_name, values) in enumerate(metrics_by_param.items()):
            ax.plot(param_values, values,
                    marker='o', linewidth=2, markersize=7,
                    color=colors[i % len(colors)],
                    label=series_name)

        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name} vs {param_name}', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = save_path or str(self.output_dir / f'scaling_{param_name}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path
