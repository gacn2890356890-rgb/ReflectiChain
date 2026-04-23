import sys
import time
sys.path.insert(0, r"c:\Users\Administrator\Desktop\cv")

from src.llm.ollama_client import create_ollama_client
client = create_ollama_client(base_url="http://localhost:11434", model="qwen:7b")

print("Measuring Ollama call latency...\n")

# Test a few calls
prompts = [
    ("action generation", "You are a supply chain planner. List 3 possible actions for procurement. Reply briefly."),
    ("internal critic", "Rate this action from 0-100 for supply chain compliance: increase_inventory. Reply with a number."),
    ("external critic", "Evaluate supply chain risk for action: reduce_shipment. Reply with risk 0-100."),
]

for name, prompt in prompts:
    t0 = time.time()
    resp = client.chat.completions.create(
        model="qwen:7b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=50,
    )
    latency = time.time() - t0
    content = resp.choices[0].message.content
    print(f"  {name}: {latency:.1f}s -> '{content[:80]}'")

print("\nEstimate: each agent step needs ~3-5 LLM calls")
print("  Main exp: 4 baselines + ReflectiChain x 5 episodes x 15 steps = ~375 calls")
print("  Each call ~15s -> ~1.5 hours just for main exp")
print("  Ablation + Scaling + MA + Adv experiments -> ~3+ hours total")
