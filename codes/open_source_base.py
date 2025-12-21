#!/usr/bin/env python3
import os
import time
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import random

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# -----------------------------
# Command-line arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Run vote prediction study with LLMs")
parser.add_argument("--model_name", type=str, required=True,
                    help="Model name or path (e.g., llama3-8b, qwen-70b)")
parser.add_argument("--data_path", type=str, required=True,
                    help="Path to JSON dataset file")
parser.add_argument("--out_dir", type=str, default="./output",
                    help="Directory to save results")
parser.add_argument("--n_samples", type=int, default=1,
                    help="Number of samples per input")
parser.add_argument("--sleep", type=float, default=0.1,
                    help="Sleep time between samples (seconds)")
args = parser.parse_args()

# -----------------------------
# Load dataset
# -----------------------------
with open(args.data_path, "r") as f:
    data = json.load(f)

random.shuffle(data)
# data = data[:3]  # for testing; remove in full run

# -----------------------------
# Candidates
# -----------------------------
candidates = ["Joe Biden", "Donald Trump"]
candidates_norm = [c.lower() for c in candidates]

os.makedirs(args.out_dir, exist_ok=True)

# -----------------------------
# Load model + tokenizer
# -----------------------------
print(f"Loading model {args.model_name} ...")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()

# -----------------------------
# Helper functions
# -----------------------------
LABEL_MAP = {
    "A": "Donald Trump",
    "B": "Joe Biden"
}

def strip_assistant_messages(messages):
    """Remove any assistant messages (ground truth leakage)."""
    return [m for m in messages if m['role'] != 'assistant']

def get_vote_probs(messages, max_new_tokens=5):
    """
    Get model probabilities for each candidate using local model.
    """
    clean_messages = strip_assistant_messages(messages)
    prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in clean_messages])

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True
        )

    # Grab first generated token
    token_id = output.sequences[0, inputs["input_ids"].shape[1]]
    token_str = tokenizer.decode(token_id).strip().lower()

    # Map token to candidate
    probs = {c: 0.0 for c in candidates}
    for c in candidates:
        if c.lower() in token_str:
            probs[c] = 1.0

    # fallback uniform
    if sum(probs.values()) == 0:
        probs = {c: 1/len(candidates) for c in candidates}

    # normalize
    Z = sum(probs.values())
    probs = {k: v/Z for k,v in probs.items()}
    return probs

def calculate_accuracy(probs, ground_truth):
    return int(max(probs, key=probs.get) == ground_truth)

def mutual_information(probs, ground_truth, eps=1e-12):
    p = max(probs.get(ground_truth, eps), eps)
    return -np.log2(p)

# -----------------------------
# Process dataset
# -----------------------------
results = []

for idx, entry in tqdm(enumerate(data), total=len(data)):
    messages = entry["messages"]

    # Ground truth from assistant
    ground_truth = next((m["content"].strip() for m in messages if m["role"]=="assistant"), None)
    if ground_truth is None:
        print(f"Warning: no assistant at index {idx}")
        continue

    if ground_truth.lower().strip() not in candidates_norm:
        print(f"Warning: unknown ground truth at index {idx}: {ground_truth}")
        continue

    probs = get_vote_probs(messages)
    acc = calculate_accuracy(probs, ground_truth)
    mi = mutual_information(probs, ground_truth)

    results.append({
        "idx": idx,
        "messages": messages,
        "probs": probs,
        "ground_truth": ground_truth,
        "accuracy": acc,
        "mutual_inf": mi
    })

    time.sleep(args.sleep)

# -----------------------------
# Save results
# -----------------------------
df_out = pd.DataFrame(results)
out_file = os.path.join(args.out_dir, f"{args.model_name.replace('/', '_')}_results.pkl")
df_out.to_pickle(out_file)
print(f"Saved predictions with metrics to {out_file}")
