#!/usr/bin/env python3
import os
import time
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    Get model probabilities for each candidate using local model logits.
    """
    clean_messages = strip_assistant_messages(messages)
    prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in clean_messages])

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model(**inputs)
        # take logits of the last token
        last_token_logits = output.logits[0, -1, :]
        probs = torch.softmax(last_token_logits, dim=-1)

    # map to candidates by checking token strings
    probs_dict = {}
    for c in candidates:
        # search for candidate token in tokenizer vocab
        token_id = tokenizer.encode(c, add_special_tokens=False)
        if len(token_id) > 0:
            prob = probs[token_id[0]].item()
        else:
            prob = 0.0
        probs_dict[c] = prob

    # fallback uniform
    if sum(probs_dict.values()) == 0:
        probs_dict = {c: 1/len(candidates) for c in candidates}

    # normalize
    Z = sum(probs_dict.values())
    probs_dict = {k: v/Z for k, v in probs_dict.items()}

    return probs_dict

# -----------------------------
# Metric functions
# -----------------------------
def calculate_accuracy_row(row):
    guess = max(row["probs"], key=row["probs"].get)
    return int(guess.lower().strip() == row["ground_truth"].lower().strip())

def mutual_information_row(row, eps=1e-12):
    p = max(row["probs"].get(row["ground_truth"], eps), eps)
    return -np.log2(p)

def agg_prob_dicts(dicts):
    n = len(dicts)
    agg_dict = {}
    for d in dicts:
        for k, v in d.items():
            agg_dict[k] = agg_dict.get(k, 0) + v / n
    return agg_dict

def ensemble(df):
    df_grouped = df.groupby("raw_idx").agg(
        {"probs": agg_prob_dicts, "ground_truth": lambda x: x.iloc[0]}
    )
    df_grouped["accuracy"] = df_grouped.apply(calculate_accuracy_row, axis=1)
    df_grouped["mutual_inf"] = df_grouped.apply(mutual_information_row, axis=1)
    df_grouped["correct_weight"] = 1.0
    return df_grouped

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

    results.append({
        "raw_idx": idx,
        "messages": messages,
        "probs": probs,
        "ground_truth": ground_truth,
        "template_name": "fixed_template",  # fake template_name for plotting
        "accuracy": calculate_accuracy_row({"probs": probs, "ground_truth": ground_truth}),
        "mutual_inf": mutual_information_row({"probs": probs, "ground_truth": ground_truth}),
        "correct_weight": 1.0
    })

    time.sleep(args.sleep)

# -----------------------------
# Ensemble metrics
# -----------------------------
df = pd.DataFrame(results)
df_ensemble = ensemble(df)

print(df)

# Save results
out_file = os.path.join(args.out_dir, f"{args.model_name.replace('/', '_')}_processed_with_metrics.pkl")
df.to_pickle(out_file)
print(f"Saved predictions with metrics to {out_file}")

# Print summary metrics
print("Average Accuracy:", df["accuracy"].mean())
print("Ensemble Accuracy:", df_ensemble["accuracy"].mean())
print("Average Mutual Info:", df["mutual_inf"].mean())



# #!/usr/bin/env python3
# import os
# import time
# import json
# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# import argparse
# import random

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # -----------------------------
# # Command-line arguments
# # -----------------------------
# parser = argparse.ArgumentParser(description="Run vote prediction study with LLMs")
# parser.add_argument("--model_name", type=str, required=True,
#                     help="Model name or path (e.g., meta-llama/Llama-3.1-8B-Instruct, qwen-70b)")
# parser.add_argument("--data_path", type=str, required=True,
#                     help="Path to JSON dataset file")
# parser.add_argument("--out_dir", type=str, default="./output",
#                     help="Directory to save results")
# parser.add_argument("--n_samples", type=int, default=1,
#                     help="Number of samples per input")
# parser.add_argument("--sleep", type=float, default=0.1,
#                     help="Sleep time between samples (seconds)")
# args = parser.parse_args()

# # -----------------------------
# # Load dataset
# # -----------------------------
# with open(args.data_path, "r") as f:
#     data = json.load(f)

# random.shuffle(data)

# # -----------------------------
# # Candidates
# # -----------------------------
# candidates = ["Donald Trump", "Joe Biden"]
# candidates_norm = [c.lower() for c in candidates]

# os.makedirs(args.out_dir, exist_ok=True)

# # -----------------------------
# # Load model + tokenizer
# # -----------------------------
# print(f"Loading model {args.model_name} ...")
# tokenizer = AutoTokenizer.from_pretrained(args.model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     args.model_name,
#     device_map="auto",
#     torch_dtype=torch.float16
# )
# model.eval()
# device = next(model.parameters()).device

# # -----------------------------
# # Helper functions
# # -----------------------------
# LABELS = ["A", "B"]  # A = Donald Trump, B = Joe Biden
# LABEL_MAP = {"A": "Donald Trump", "B": "Joe Biden"}
# label_ids = [tokenizer(label, return_tensors="pt").input_ids[0] for label in LABELS]

# def strip_assistant_messages(messages):
#     """Remove any assistant messages (ground truth leakage)."""
#     return [m for m in messages if m['role'] != 'assistant']

# def get_vote_probs(messages):
#     """
#     Get model probabilities for each candidate using single-token logits.
#     """
#     clean_messages = strip_assistant_messages(messages)
#     prompt = "\n".join([f"{m['role']}: {m['content']}" for m in clean_messages])

#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits[:, -1, :]  # last token logits

#         probs = {}
#         for label, ids in zip(LABELS, label_ids):
#             probs[LABEL_MAP[label]] = torch.softmax(logits[:, ids[0]], dim=-1).item()

#     # Normalize
#     Z = sum(probs.values())
#     probs = {k: v/Z for k, v in probs.items()}
#     return probs

# def calculate_accuracy(probs, ground_truth):
#     return int(max(probs, key=probs.get) == ground_truth)

# def mutual_information(probs, ground_truth, eps=1e-12):
#     p = max(probs.get(ground_truth, eps), eps)
#     return -np.log2(p)

# # -----------------------------
# # Process dataset
# # -----------------------------
# results = []

# for idx, entry in tqdm(enumerate(data), total=len(data)):
#     messages = entry["messages"]

#     # Extract ground truth from assistant
#     ground_truth = next((m["content"].strip() for m in messages if m["role"]=="assistant"), None)
#     if ground_truth is None:
#         print(f"Warning: no assistant at index {idx}")
#         continue
#     if ground_truth.lower() not in candidates_norm:
#         print(f"Warning: unknown ground truth at index {idx}: {ground_truth}")
#         continue

#     probs = get_vote_probs(messages)
#     acc = calculate_accuracy(probs, ground_truth)
#     mi = mutual_information(probs, ground_truth)

#     results.append({
#         "idx": idx,
#         "messages": messages,
#         "probs": probs,
#         "ground_truth": ground_truth,
#         "accuracy": acc,
#         "mutual_inf": mi
#     })

#     time.sleep(args.sleep)

# # -----------------------------
# # Save results
# # -----------------------------
# df_out = pd.DataFrame(results)
# print(df_out)
# out_file = os.path.join(args.out_dir, f"{args.model_name.replace('/', '_')}_results.pkl")
# df_out.to_pickle(out_file)
# print(f"Saved predictions with metrics to {out_file}")
