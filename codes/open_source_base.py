# #!/usr/bin/env python3

# import os
# import time
# import json
# import argparse
# import random
# import numpy as np
# import pandas as pd
# import torch
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from sklearn.metrics import cohen_kappa_score

# # Optional package
# try:
#     import pingouin as pg
#     icc_available = True
# except ImportError:
#     print("pingouin not installed, ICC will be skipped")
#     icc_available = False

# # ==========================================
# # Command-line arguments
# # ==========================================

# parser = argparse.ArgumentParser(description="Vote prediction with LLMs")
# parser.add_argument("--model_name", type=str, required=True,
#                     help="HF model name or local path")
# parser.add_argument("--data_path", type=str, required=True,
#                     help="Path to input JSON dataset")
# parser.add_argument("--out_dir", type=str, default="./output",
#                     help="Directory to save results")
# parser.add_argument("--sleep", type=float, default=0.1,
#                     help="Sleep time between requests")
# args = parser.parse_args()

# os.makedirs(args.out_dir, exist_ok=True)

# # ==========================================
# # Constants
# # ==========================================

# CANDIDATES = ["Joe Biden", "Donald Trump"]
# CANDIDATES_NORM = [c.lower() for c in CANDIDATES]
# SEED = 42

# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)

# # ==========================================
# # Load dataset
# # ==========================================

# with open(args.data_path, "r") as f:
#     data = json.load(f)

# df_primary = pd.DataFrame(data)
# df_primary['raw_idx'] = list(range(len(data)))

# # Build input dataframe
# df_input = pd.DataFrame({
#     "raw_idx": list(range(len(data))),
#     "messages": [entry["messages"] for entry in data]
# })

# print(f"Loaded {len(df_input)} samples")

# # ==========================================
# # Load model & tokenizer
# # ==========================================

# print(f"Loading model: {args.model_name}")

# tokenizer = AutoTokenizer.from_pretrained(args.model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     args.model_name,
#     device_map="auto",
#     torch_dtype=torch.float16
# )
# model.eval()
# device = next(model.parameters()).device

# # ==========================================
# # Helper functions
# # ==========================================

# def strip_assistant_messages(messages):
#     return [m for m in messages if m["role"] != "assistant"]

# def extract_ground_truth(messages):
#     for m in messages:
#         if m["role"] == "assistant":
#             return m["content"].strip()
#     return None

# def get_vote_probs(messages):
#     clean_msgs = strip_assistant_messages(messages)
#     prompt = "\n".join(f"{m['role']}: {m['content']}" for m in clean_msgs)

#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits[0, -1, :]
#         probs = torch.softmax(logits, dim=-1)

#     cand_probs = {}
#     for c in CANDIDATES:
#         token_ids = tokenizer.encode(c, add_special_tokens=False)
#         cand_probs[c] = probs[token_ids[0]].item() if token_ids else 0.0

#     if sum(cand_probs.values()) == 0:
#         cand_probs = {c: 1 / len(CANDIDATES) for c in CANDIDATES}

#     Z = sum(cand_probs.values())
#     return {k: v / Z for k, v in cand_probs.items()}

# def accuracy_from_probs(probs, ground_truth):
#     pred = max(probs, key=probs.get)
#     return int(pred.lower() == ground_truth.lower())

# def mutual_information(probs, ground_truth, eps=1e-12):
#     p = max(probs.get(ground_truth, eps), eps)
#     return -np.log2(p)

# # ==========================================
# # Inference loop
# # ==========================================

# rows = []

# for idx, row in tqdm(df_input.iterrows(), total=len(df_input)):
#     messages = row["messages"]
#     ground_truth = extract_ground_truth(messages)
#     if ground_truth is None:
#         continue
#     if ground_truth.lower() not in CANDIDATES_NORM:
#         continue

#     probs = get_vote_probs(messages)
#     acc = accuracy_from_probs(probs, ground_truth)
#     mi = mutual_information(probs, ground_truth)

#     predicted_vote = max(probs, key=probs.get)

#     rows.append({
#         "raw_idx": row["raw_idx"],
#         "probs": probs,
#         "predicted_vote": predicted_vote,
#         "ground_truth": ground_truth,
#         "accuracy": acc,
#         "mutual_inf": mi,
#         "template_name": "fixed_template",
#         "correct_weight": 1.0,
#         "model_name": args.model_name
#     })

#     time.sleep(args.sleep)

# df_results = pd.DataFrame(rows)
# print("Inference complete")
# print(df_results[["accuracy", "mutual_inf"]].describe())

# # ==========================================
# # Merge with input dataframe
# # ==========================================

# df_final = df_primary.merge(
#     df_results,
#     on="raw_idx",
#     how="left"
# )

# # ==========================================
# # Compute vote correspondence metrics
# # ==========================================

# # Convert votes to numeric 0/1
# def vote_to_numeric(vote):
#     return 0 if vote.lower() == "joe biden" else 1

# anes_votes = df_final['ground_truth'].dropna().map(vote_to_numeric).to_numpy()
# gpt_votes = df_final['predicted_vote'].dropna().map(vote_to_numeric).to_numpy()

# vote_metrics = {}

# # Cohen's Kappa
# vote_metrics['cohen_kappa'] = cohen_kappa_score(anes_votes, gpt_votes)

# # ICC
# if icc_available:
#     try:
#         df_temp = pd.DataFrame({'anes': anes_votes, 'gpt': gpt_votes})
#         df_long = df_temp.reset_index().melt(id_vars='index', value_vars=['anes','gpt'],
#                                             var_name='rater', value_name='vote')
#         icc_df = pg.intraclass_corr(data=df_long, targets='index', raters='rater', ratings='vote')
#         icc_value = icc_df.loc[icc_df['Type']=='ICC2k','ICC'].values[0]
#         vote_metrics['ICC'] = icc_value
#     except Exception as e:
#         print(f"Warning: could not compute ICC: {e}")
#         vote_metrics['ICC'] = None
# else:
#     vote_metrics['ICC'] = None

# # Proportion agreement
# vote_metrics['proportion_agreement'] = np.mean(anes_votes == gpt_votes)

# # Add metrics to dataframe
# for k,v in vote_metrics.items():
#     df_final[k] = v

# # ==========================================
# # Save final dataframe
# # ==========================================

# base_name = args.model_name.replace("/", "_")
# pkl_path = os.path.join(args.out_dir, f"{base_name}_final.pkl")
# csv_path = os.path.join(args.out_dir, f"{base_name}_final.csv")

# df_final.to_pickle(pkl_path)
# df_final.to_csv(csv_path, index=False)

# print(f"Saved final dataframe with metrics:")
# print(f" - {pkl_path}")
# print(f" - {csv_path}")
# print(df_final.head())

# # ==========================================
# # Summary
# # ==========================================

# print("\nSummary:")
# print("Average accuracy:", df_results["accuracy"].mean())
# print("Average mutual information:", df_results["mutual_inf"].mean())
# print("Vote correspondence metrics:")
# for k,v in vote_metrics.items():
#     print(f"{k}: {v}")


#!/usr/bin/env python3

import os
import time
import json
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import cohen_kappa_score

# Optional ICC
try:
    import pingouin as pg
    icc_available = True
except ImportError:
    print("pingouin not installed, ICC will be skipped")
    icc_available = False

# -----------------------------
# Arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Vote prediction study with LLMs")
parser.add_argument("--model_name", type=str, required=True,
                    help="HF model name or local path")
parser.add_argument("--data_path", type=str, required=True,
                    help="Path to JSON dataset")
parser.add_argument("--out_dir", type=str, default="./output",
                    help="Output directory")
parser.add_argument("--election_year", type=int, choices=[2020, 2024], required=True,
                    help="Election year")
parser.add_argument("--sleep", type=float, default=0.1,
                    help="Sleep time between samples")
parser.add_argument("--save_every", type=int, default=1000,
                    help="Save intermediate results every N samples")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# -----------------------------
# Candidates
# -----------------------------
if args.election_year == 2020:
    CANDIDATES = ["Donald Trump", "Joe Biden"]
elif args.election_year == 2024:
    CANDIDATES = ["Donald Trump", "Kamala Harris"]
CANDIDATES_NORM = [c.lower() for c in CANDIDATES]

# -----------------------------
# Load dataset
# -----------------------------
with open(args.data_path, "r") as f:
    data = json.load(f)
random.shuffle(data)
print(f"Loaded {len(data)} samples")

# -----------------------------
# Load model
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
def strip_assistant_messages(messages):
    return [m for m in messages if m["role"] != "assistant"]

def normalize_vote(text):
    if text is None: return None
    t = text.lower().strip()
    if "trump" in t: return "Donald Trump"
    if "donald" in t: return "Donald Trump"
    if "biden" in t: return "Joe Biden"
    if "joe" in t: return "Joe Biden"
    if "harris" in t: return "Kamala Harris"
    if "kamala" in t: return "Kamala Harris"
        
    return None

def extract_ground_truth(messages):
    for m in messages:
        if m["role"] == "assistant":
            return normalize_vote(m["content"])
    return None

def get_vote_probs(messages, max_new_tokens=10):
    """
    Generate vote probabilities for the given messages.
    Handles multi-token candidate names and normalizes output.
    """
    clean_msgs = strip_assistant_messages(messages)
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in clean_msgs)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # deterministic
            output_scores=False,
            return_dict_in_generate=True
        )

    # decode all generated tokens
    token_ids = output.sequences[0, inputs["input_ids"].shape[1]:]
    token_str = tokenizer.decode(token_ids).lower()

    # normalize candidate
    predicted_vote = normalize_vote(token_str)

    # fallback: uniform if model output not recognized
    probs = {c: 0.0 for c in CANDIDATES}
    if predicted_vote in CANDIDATES:
        probs[predicted_vote] = 1.0
    else:
        probs = {c: 1 / len(CANDIDATES) for c in CANDIDATES}

    return probs


def accuracy_from_probs(probs, ground_truth):
    return int(max(probs, key=probs.get) == ground_truth)

def mutual_information(probs, ground_truth, eps=1e-12):
    p = max(probs.get(ground_truth, eps), eps)
    return -np.log2(p)

def vote_to_numeric(vote):
    return 0 if vote.lower() == CANDIDATES[1].lower() else 1

# -----------------------------
# Inference loop
# -----------------------------
results = []
for idx, entry in tqdm(enumerate(data), total=len(data)):
    messages = entry.get("messages", [])
    gt = extract_ground_truth(messages)
    if gt is None or gt.lower() not in CANDIDATES_NORM:
        continue

    probs = get_vote_probs(messages)
    acc = accuracy_from_probs(probs, gt)
    mi = mutual_information(probs, gt)
    pred = max(probs, key=probs.get)

    results.append({
        "idx": idx,
        "messages": messages,
        "ground_truth": gt,
        "predicted_vote": pred,
        "probs": probs,
        "accuracy": acc,
        "mutual_inf": mi
    })

    if (idx+1) % args.save_every == 0:
        df_tmp = pd.DataFrame(results)
        save_path = os.path.join(args.out_dir, f"{args.model_name.replace('/', '_')}_{args.election_year}_partial.pkl")
        df_tmp.to_pickle(save_path)
        print(f"Saved intermediate results at index {idx} to {save_path}")

    time.sleep(args.sleep)

df_final = pd.DataFrame(results)

# -----------------------------
# Compute vote correspondence metrics
# -----------------------------
anes_votes = df_final['ground_truth'].map(vote_to_numeric).to_numpy()
gpt_votes = df_final['predicted_vote'].map(vote_to_numeric).to_numpy()

vote_metrics = {}
vote_metrics['cohen_kappa'] = cohen_kappa_score(anes_votes, gpt_votes)

if icc_available:
    try:
        df_temp = pd.DataFrame({'anes': anes_votes, 'gpt': gpt_votes})
        df_long = df_temp.reset_index().melt(id_vars='index', value_vars=['anes','gpt'],
                                            var_name='rater', value_name='vote')
        icc_df = pg.intraclass_corr(data=df_long, targets='index', raters='rater', ratings='vote')
        vote_metrics['ICC'] = icc_df.loc[icc_df['Type']=='ICC2k','ICC'].values[0]
    except Exception as e:
        print(f"Could not compute ICC: {e}")
        vote_metrics['ICC'] = None
else:
    vote_metrics['ICC'] = None

vote_metrics['proportion_agreement'] = np.mean(anes_votes == gpt_votes)

for k,v in vote_metrics.items():
    df_final[k] = v

# -----------------------------
# Save final results
# -----------------------------
out_file = os.path.join(args.out_dir, f"{args.model_name.replace('/', '_')}_{args.election_year}_final.pkl")
df_final.to_pickle(out_file)
df_final.to_csv(out_file.replace(".pkl",".csv"), index=False)
print(f"Saved final results to {out_file}")

# -----------------------------
# Summary
# -----------------------------
print("\nSummary:")
print("Average accuracy:", df_final["accuracy"].mean())
print("Average mutual information:", df_final["mutual_inf"].mean())
for k,v in vote_metrics.items():
    print(f"{k}: {v}")

