import torch
import numpy as np
from kv_gym.vendor.loader import load_model_and_tokenizer
from kv_gym.vendor.gsm8k import load_gsm8k

def main():
    print("Loading model and tokenizer...")
    model, tokenizer, device = load_model_and_tokenizer("qwen-1.5b")
    
    print("Loading test dataset...")
    examples = load_gsm8k(n=100, seed=42, split="test")
    
    gen_lengths = []
    truncated_128 = 0
    truncated_256 = 0
    truncated_384 = 0
    
    print("Starting generation...")
    for i, ex in enumerate(examples):
        inputs = tokenizer(ex["prompt_text"], return_tensors="pt").to(device)
        T = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            out = model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=512,
                do_sample=False,
            )
        
        gen_len = out.shape[1] - T
        gen_lengths.append(gen_len)
        
        if gen_len >= 128:
            truncated_128 += 1
        if gen_len >= 256:
            truncated_256 += 1
        if gen_len >= 384:
            truncated_384 += 1
            
        print(f"Ex {i:3d}: T_prompt={T:3d}, T_gen={gen_len:3d}")
        
    print("\n=== Generation Statistics ===")
    print(f"Total examples: {len(examples)}")
    print(f"Mean generation length: {np.mean(gen_lengths):.1f}")
    print(f"Median generation length: {np.median(gen_lengths):.1f}")
    print(f"Max generation length: {np.max(gen_lengths)}")
    print(f"Min generation length: {np.min(gen_lengths)}")
    print(f"Truncated at 128: {truncated_128} ({truncated_128 / len(examples):.1%})")
    print(f"Truncated at 256: {truncated_256} ({truncated_256 / len(examples):.1%})")
    print(f"Truncated at 384: {truncated_384} ({truncated_384 / len(examples):.1%})")

if __name__ == "__main__":
    main()
