#!/usr/bin/env python3

import os
import time
import json
import argparse
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import cohen_kappa_score

# Optional packages
try:
    from factor_analyzer import calculate_tetrachoric
    tetra_available = True
except ImportError:
    print("factor_analyzer not installed, tetrachoric correlation will be skipped")
    tetra_available = False

try:
    import pingouin as pg
    icc_available = True
except ImportError:
    print("pingouin not installed, ICC will be skipped")
    icc_available = False

# ==========================================
# Command-line arguments
# ==========================================

parser = argparse.ArgumentParser(description="Vote prediction with LLMs")
parser.add_argument("--model_name", type=str, required=True,
                    help="HF model name or local path")
parser.add_argument("--data_path", type=str, required=True,
                    help="Path to input JSON dataset")
parser.add_argument("--out_dir", type=str, default="./output",
                    help="Directory to save results")
parser.add_argument("--sleep", type=float, default=0.1,
                    help="Sleep time between requests")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ==========================================
# Constants
# ==========================================

CANDIDATES = ["Joe Biden", "Donald Trump"]
CANDIDATES_NORM = [c.lower() for c in CANDIDATES]
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ==========================================
# Load dataset
# ==========================================

with open(args.data_path, "r") as f:
    data = json.load(f)

df_primary = pd.DataFrame(data)
df_primary['raw_idx'] = list(range(len(data)))

# Build input dataframe
df_input = pd.DataFrame({
    "raw_idx": list(range(len(data))),
    "messages": [entry["messages"] for entry in data]
})

print(f"Loaded {len(df_input)} samples")

# ==========================================
# Load model & tokenizer
# ==========================================

print(f"Loading model: {args.model_name}")

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()
device = next(model.parameters()).device

# ==========================================
# Helper functions
# ==========================================

def strip_assistant_messages(messages):
    return [m for m in messages if m["role"] != "assistant"]

def extract_ground_truth(messages):
    for m in messages:
        if m["role"] == "assistant":
            return m["content"].strip()
    return None

def get_vote_probs(messages):
    clean_msgs = strip_assistant_messages(messages)
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in clean_msgs)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

    cand_probs = {}
    for c in CANDIDATES:
        token_ids = tokenizer.encode(c, add_special_tokens=False)
        cand_probs[c] = probs[token_ids[0]].item() if token_ids else 0.0

    if sum(cand_probs.values()) == 0:
        cand_probs = {c: 1 / len(CANDIDATES) for c in CANDIDATES}

    Z = sum(cand_probs.values())
    return {k: v / Z for k, v in cand_probs.items()}

def accuracy_from_probs(probs, ground_truth):
    pred = max(probs, key=probs.get)
    return int(pred.lower() == ground_truth.lower())

def mutual_information(probs, ground_truth, eps=1e-12):
    p = max(probs.get(ground_truth, eps), eps)
    return -np.log2(p)

# ==========================================
# Inference loop
# ==========================================

rows = []

for idx, row in tqdm(df_input.iterrows(), total=len(df_input)):
    messages = row["messages"]
    ground_truth = extract_ground_truth(messages)
    if ground_truth is None:
        continue
    if ground_truth.lower() not in CANDIDATES_NORM:
        continue

    probs = get_vote_probs(messages)
    acc = accuracy_from_probs(probs, ground_truth)
    mi = mutual_information(probs, ground_truth)

    predicted_vote = max(probs, key=probs.get)

    rows.append({
        "raw_idx": row["raw_idx"],
        "probs": probs,
        "predicted_vote": predicted_vote,
        "ground_truth": ground_truth,
        "accuracy": acc,
        "mutual_inf": mi,
        "template_name": "fixed_template",
        "correct_weight": 1.0,
        "model_name": args.model_name
    })

    time.sleep(args.sleep)

df_results = pd.DataFrame(rows)
print("Inference complete")
print(df_results[["accuracy", "mutual_inf"]].describe())

# ==========================================
# Merge with input dataframe
# ==========================================

df_final = df_primary.merge(
    df_results,
    on="raw_idx",
    how="left"
)

# ==========================================
# Compute vote correspondence metrics
# ==========================================

# Convert votes to numeric 0/1 for tetrachoric
def vote_to_numeric(vote):
    return 0 if vote.lower() == "joe biden" else 1

anes_votes = df_final['ground_truth'].dropna().map(vote_to_numeric).to_numpy()
gpt_votes = df_final['predicted_vote'].dropna().map(vote_to_numeric).to_numpy()

vote_metrics = {}

# Tetrachoric correlation
try:
    if tetra_available and len(np.unique(anes_votes)) > 1 and len(np.unique(gpt_votes)) > 1:
        df_votes = pd.DataFrame({'anes': anes_votes, 'gpt': gpt_votes})
        tetra_corr = calculate_tetrachoric(df_votes)
        vote_metrics['tetrachoric'] = tetra_corr[0,1]
    else:
        vote_metrics['tetrachoric'] = None
except Exception as e:
    print(f"Warning: could not compute tetrachoric correlation: {e}")
    vote_metrics['tetrachoric'] = None

# Cohen's Kappa
vote_metrics['cohen_kappa'] = cohen_kappa_score(anes_votes, gpt_votes)

# ICC
try:
    if icc_available and len(np.unique(anes_votes)) > 1 and len(np.unique(gpt_votes)) > 1:
        df_temp = pd.DataFrame({'anes': anes_votes, 'gpt': gpt_votes})
        df_long = df_temp.reset_index().melt(id_vars='index', value_vars=['anes','gpt'],
                                            var_name='rater', value_name='vote')
        icc_df = pg.intraclass_corr(data=df_long, targets='index', raters='rater', ratings='vote')
        icc_value = icc_df.loc[icc_df['Type']=='ICC2k','ICC'].values[0]
        vote_metrics['ICC'] = icc_value
    else:
        vote_metrics['ICC'] = None
except Exception as e:
    print(f"Warning: could not compute ICC: {e}")
    vote_metrics['ICC'] = None

# Proportion agreement
vote_metrics['proportion_agreement'] = np.mean(anes_votes == gpt_votes)

# Add metrics to dataframe for saving
for k, v in vote_metrics.items():
    df_final[k] = v

# ==========================================
# Save final dataframe
# ==========================================

base_name = args.model_name.replace("/", "_")
pkl_path = os.path.join(args.out_dir, f"{base_name}_final.pkl")
csv_path = os.path.join(args.out_dir, f"{base_name}_final.csv")

df_final.to_pickle(pkl_path)
df_final.to_csv(csv_path, index=False)

print(f"Saved final dataframe with metrics:")
print(f" - {pkl_path}")
print(f" - {csv_path}")
print(df_final.head())

# ==========================================
# Summary
# ==========================================

print("\nSummary:")
print("Average accuracy:", df_results["accuracy"].mean())
print("Average mutual information:", df_results["mutual_inf"].mean())
print("Vote correspondence metrics:")
for k,v in vote_metrics.items():
    print(f"{k}: {v}")
