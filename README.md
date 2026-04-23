# ReflectiChain: Agentic World Model for Semiconductor Supply Chain Resilience

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2604.11041-b31b1b.svg)](https://arxiv.org/abs/2604.11041)
[![Conference](https://img.shields.io/badge/Conference-Under_Review-blue.svg)](#) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ollama Supported](https://img.shields.io/badge/LLM-Ollama_Local-orange.svg)](https://ollama.ai/)

This repository contains the official code implementation for the paper **"ReflectiChain: Agentic World Model Framework for Semiconductor Supply Chain Resilience and Policy-Aware Planning"**.

</div>

## 📖 Abstract

Due to the high cost of trial-and-error, complex globalized networks, and long-cycle characteristics, the semiconductor supply chain is highly sensitive to policy fluctuations and external shocks. **ReflectiChain** proposes a novel framework based on an Agentic World Model, aiming to achieve highly resilient, Policy-Aware Planning.

We introduce two core mechanisms in this framework:
1. **Latent Trajectory Rehearsal**: Agents can "rehearse" supply chain dynamics under various policy shocks or chain disruptions within the latent space of the world model, optimizing strategies in a risk-free virtual environment.
2. **Retrospective Agentic Reinforcement Learning**: Empowers agents with the ability to reflect and correct based on historical sub-optimal decisions and supply chain collapse events, significantly enhancing robustness against long-tail risks.

## 📂 Repository Structure

The codebase is organized as follows to ensure the reproducibility of the experiments:

```text
ReflectiChain/
├── src/                          # Core framework source code of ReflectiChain
│   ├── env/                      # Semiconductor supply chain simulation environment
│   ├── world_model/              # Latent trajectory rehearsal module implementation
│   └── agent/                    # Retrospective reinforcement learning agent
├── config.py                     # Global experiment and hyperparameter configurations
├── run_exp.py                    # Main experiment execution script
├── run_ma_adv.py                 # Multi-Agent Adversarial execution script
├── gen_ma_adv.py                 # Multi-agent adversarial scenario generator
├── test_ollama.py                # Connectivity and inference test for local LLM (Ollama)
├── test_latency.py               # Inference latency and system overhead test
├── generate_paper_results.py     # One-click script to reproduce core data and tables
├── gen_fig.py                    # Paper figure and chart visualization script
└── requirements.txt              # Environment dependencies
```
## 🛠️ Installation

1. Clone this repository:

Bash

```
git clone [https://github.com/gacn2890356890-rgb/ReflectiChain.git](https://github.com/gacn2890356890-rgb/ReflectiChain.git)
cd ReflectiChain
```

1. Create and activate a virtual environment (Conda is recommended):

Bash

```
conda create -n reflectichain python=3.10
conda activate reflectichain
```

1. Install dependencies:

Bash

```
pip install -r requirements.txt
```

1. **Configure Ollama (Optional but recommended)**: To protect data privacy and reduce latency, this framework supports invoking local large language models via Ollama as the cognitive engine for the Agents. Please ensure the Ollama service is running locally. You can run the following script to test the connection:

Bash

```
python test_ollama.py
```

## 🚀 Quick Start

You can launch a single supply chain planning rehearsal based on the default configuration using `run_exp.py`:

Bash

```
python run_exp.py --config config.py --mode train
```

To verify multi-agent interactions and resilience performance under adversarial policy interventions, please run:

Bash

```
# 1. Generate multi-agent adversarial scenarios
python gen_ma_adv.py --scenario "trade_restriction" 

# 2. Run adversarial rehearsal
python run_ma_adv.py --scenario_path ./data/adv_scenario.json
```

## 📊 Reproducing Paper Results

We provide one-click scripts to fully reproduce the experimental results and figures in the paper. Please ensure you have sufficient computational resources (a GPU with at least 24GB VRAM is recommended to accelerate world model inference).

1. **Generate Experimental Data**: This script will automatically run all baselines and generate the required metric data (saved in the `results/` directory):

Bash

```
python generate_paper_results.py
```

1. **Generate Paper Figures**: Based on the generated data, plot all main figures and appendix charts for the paper (output to the `figures/` directory):

Bash

```
python gen_fig.py
```

1. **System Latency Evaluation** (corresponding to the ablation study/deployment feasibility analysis in the paper):

Bash

```
python test_latency.py
```

## 📝 Citation

If you use our code or world model framework in your research, please cite our paper:

Code snippet

```
@misc{luo2026topologytrajectoryllmdrivenworld,
      title={From Topology to Trajectory: LLM-Driven World Models For Supply Chain Resilience}, 
      author={Jia Luo},
      year={2026},
      eprint={2604.11041},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2604.11041}, 
}
```

## 📄 License

This project is open-sourced under the [MIT License](https://www.google.com/search?q=LICENSE).
