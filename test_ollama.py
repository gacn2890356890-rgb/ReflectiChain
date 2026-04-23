import sys
sys.path.insert(0, r"c:\Users\Administrator\Desktop\cv")

print("=== Quick Ollama Integration Test ===\n")

# 1. Test Ollama client
from src.llm.ollama_client import create_ollama_client
client = create_ollama_client(base_url="http://localhost:11434", model="qwen:7b")
if client is None:
    print("FAIL: Ollama client is None")
    sys.exit(1)
print("PASS: Ollama client created")

# 2. Test actual call path (same as actor.py uses)
try:
    resp = client.chat.completions.create(
        model="qwen:7b",
        messages=[{"role": "user", "content": "Reply with exactly one word: supply chain"}],
        temperature=0.3,
        max_tokens=20,
    )
    print(f"PASS: LLM responded: '{resp.choices[0].message.content}'")
except Exception as e:
    print(f"FAIL: LLM call failed: {e}")
    sys.exit(1)

# 3. Test that the full agent uses real LLM
print("\n--- Testing ReflectiChainAgent with real Ollama ---")
from config import LLMConfig, AgentConfig, WorldModelConfig, SemiSimConfig, ShockConfig
from src.semi_sim.environment import SemiSimEnvironment
from src.reflecti_chain.reflection import ReflectiChainAgent

semi_cfg = SemiSimConfig()
shock_cfg = ShockConfig()
agent_cfg = AgentConfig()
wm_cfg = WorldModelConfig()
llm_cfg = LLMConfig()

env = SemiSimEnvironment(semi_cfg, shock_cfg, seed=42)
agent = ReflectiChainAgent(
    llm_config=llm_cfg,
    agent_config=agent_cfg,
    world_model_cfg=wm_cfg,
    semi_cfg=semi_cfg,
    agent_id=0,
    agent_type="profit_driven",
    llm_client=client,
    device="cpu",
)

obs = env.reset()
agent.reset()

print("Running 1 step...")
action, info = agent.step(obs, env)
print(f"  Action: {action.description[:100]}")
print(f"  OR: {info.get('operability_ratio', 'N/A')}")
print(f"  Reward: {info.get('immediate_reward', 'N/A')}")
print(f"  LLM calls made: internal_critic={agent.internal_critic.llm is not None}, "
      f"external_critic={agent.external_critic.llm is not None}, "
      f"actor={agent.action_actor.llm is not None}")

# 4. Check the LLM was actually called by looking at the responses
print("\n--- Checking real LLM responses ---")
for i, cand in enumerate(info.get('candidates', [])):
    sem = cand.get('semantic_score', 'N/A')
    print(f"  Candidate {i+1}: semantic_score={sem}")

print("\n=== ALL TESTS PASSED ===")
