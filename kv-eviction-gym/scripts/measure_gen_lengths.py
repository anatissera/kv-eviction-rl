"""Measure full-cache generation lengths for Qwen-1.5B on GSM8K."""
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from kv_gym.vendor.gsm8k import load_gsm8k
from kv_gym.vendor.prompts import format_gsm8k

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B-Instruct", torch_dtype=torch.bfloat16
).cuda()
model.eval()

examples = load_gsm8k(n=50, seed=0, split="train")
gen_lengths, prompt_lengths, correct = [], [], []
eos = tok.eos_token_id

for i, ex in enumerate(examples):
    prompt, ans = format_gsm8k(ex)
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    prompt_lengths.append(ids.shape[1])
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=800, do_sample=False, eos_token_id=eos)
    gen = out[0, ids.shape[1]:]
    gen_lengths.append(len(gen))
    text = tok.decode(gen)
    correct.append(str(ans) in text)
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/50 done")

g = np.array(gen_lengths)
p = np.array(prompt_lengths)
print(f"\nprompt:   min={p.min()} mean={p.mean():.0f} max={p.max()}")
print(f"gen_len:  min={g.min()} p25={np.percentile(g,25):.0f} p50={np.percentile(g,50):.0f}"
      f" p75={np.percentile(g,75):.0f} p90={np.percentile(g,90):.0f} max={g.max()}")
print(f"eos_rate: {(g < 800).mean():.1%}  correct: {np.mean(correct):.1%}")
print(f"\nbudget needed (prompt+gen) to reach eviction zone:")
print(f"  p50={int(p.mean()+np.percentile(g,50))}  p75={int(p.mean()+np.percentile(g,75))}"
      f"  p90={int(p.mean()+np.percentile(g,90))}")
