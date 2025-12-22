# #!/usr/bin/env python3

# import os
# import time
# import json
# import argparse
# import random
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from sklearn.metrics import cohen_kappa_score
# import openai

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
# parser = argparse.ArgumentParser(description="Vote prediction with GPT API")
# parser.add_argument("--model_name", type=str, required=True,
#                     help="OpenAI GPT model name (e.g., gpt-4, gpt-3.5-turbo)")
# parser.add_argument("--data_path", type=str, required=True,
#                     help="Path to input JSON dataset")
# parser.add_argument("--out_dir", type=str, default="./output",
#                     help="Directory to save results")
# parser.add_argument("--sleep", type=float, default=0.1,
#                     help="Sleep time between API requests")
# parser.add_argument("--api_key", type=str, required=True,
#                     help="OpenAI API key")
# args = parser.parse_args()

# os.makedirs(args.out_dir, exist_ok=True)

# # Set OpenAI API key
# openai.api_key = args.api_key


# os.makedirs(args.out_dir, exist_ok=True)

# # ==========================================
# # Constants
# # ==========================================
# CANDIDATES = ["Joe Biden", "Donald Trump"]
# CANDIDATES_NORM = [c.lower() for c in CANDIDATES]
# SEED = 42

# random.seed(SEED)
# np.random.seed(SEED)

# # ==========================================
# # Load dataset
# # ==========================================
# with open(args.data_path, "r") as f:
#     data = json.load(f)

# df_primary = pd.DataFrame(data)
# df_primary['raw_idx'] = list(range(len(data)))

# df_input = pd.DataFrame({
#     "raw_idx": list(range(len(data))),
#     "messages": [entry["messages"] for entry in data]
# })

# print(f"Loaded {len(df_input)} samples")

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

# def get_vote_probs_gpt(clean_messages, model_name):
#     """Call OpenAI API and compute probabilities for each candidate."""
#     # try:
#     response = openai.chat.completions.create(
#         model=model_name,
#         messages=clean_messages,
#         temperature=0,
#     )
#     content = response.choices[0].message.content
    
#     # except Exception as e:
#     #     print(f"Warning: GPT API call failed: {e}")
#     #     content = ""
    
#     # Simple normalization: assign 1.0 to the predicted candidate
#     probs = {c: 0.0 for c in CANDIDATES}
#     for c in CANDIDATES:
#         if c.lower() in content:
#             probs[c] = 1.0
#     # fallback uniform
#     if sum(probs.values()) == 0:
#         probs = {c: 1/len(CANDIDATES) for c in CANDIDATES}
#     # normalize
#     Z = sum(probs.values())
#     return {k: v / Z for k, v in probs.items()}

# def accuracy_from_probs(probs, ground_truth):
#     pred = max(probs, key=probs.get)
#     return int(pred.lower() == ground_truth.lower())

# def mutual_information(probs, ground_truth, eps=1e-12):
#     p = max(probs.get(ground_truth, eps), eps)
#     return -np.log2(p)

# def vote_to_numeric(vote):
#     return 0 if vote.lower() == "joe biden" else 1

# # ==========================================
# # Inference loop
# # ==========================================
# rows = []

# for idx, row in tqdm(df_input.iterrows(), total=len(df_input)):
#     messages = row["messages"]
#     ground_truth = extract_ground_truth(messages)
#     if ground_truth is None or ground_truth.lower() not in CANDIDATES_NORM:
#         continue

#     clean_messages = strip_assistant_messages(messages)
#     # Convert to GPT API format
#     chat_messages = [{"role": m["role"], "content": m["content"]} for m in clean_messages]

#     probs = get_vote_probs_gpt(chat_messages, args.model_name)
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
#         vote_metrics['ICC'] = icc_df.loc[icc_df['Type']=='ICC2k','ICC'].values[0]
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
from sklearn.metrics import cohen_kappa_score
import openai

# Optional package
try:
    import pingouin as pg
    icc_available = True
except ImportError:
    icc_available = False

# ==========================================
# Arguments
# ==========================================
parser = argparse.ArgumentParser(description="Vote prediction with GPT API (resumable)")
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--out_dir", type=str, default="./output")
parser.add_argument("--sleep", type=float, default=0.1)
parser.add_argument("--api_key", type=str, required=True)
parser.add_argument("--save_every", type=int, default=1000,
                    help="Checkpoint every N samples")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# OpenAI key
openai.api_key = args.api_key

# ==========================================
# Constants
# ==========================================
CANDIDATES = ["Joe Biden", "Donald Trump"]
CANDIDATES_NORM = [c.lower() for c in CANDIDATES]
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# ==========================================
# Load data
# ==========================================
with open(args.data_path, "r") as f:
    data = json.load(f)

df_primary = pd.DataFrame(data)
df_primary["raw_idx"] = range(len(data))

df_input = pd.DataFrame({
    "raw_idx": range(len(data)),
    "messages": [d["messages"] for d in data]
})

print(f"Loaded {len(df_input)} samples")

# ==========================================
# Checkpoint logic
# ==========================================
checkpoint_path = os.path.join(args.out_dir, "checkpoint.csv")

if os.path.exists(checkpoint_path):
    df_checkpoint = pd.read_csv(checkpoint_path)
    rows = df_checkpoint.to_dict("records")
    processed = set(df_checkpoint["raw_idx"])
    print(f"Resuming from checkpoint: {len(processed)} samples done")
else:
    rows = []
    processed = set()

# ==========================================
# Helpers
# ==========================================
def strip_assistant_messages(messages):
    return [m for m in messages if m["role"] != "assistant"]

def extract_ground_truth(messages):
    for m in messages:
        if m["role"] == "assistant":
            return m["content"].strip()
    return None

def vote_to_numeric(v):
    return 0 if v.lower() == "joe biden" else 1

def accuracy_from_probs(probs, gt):
    return int(max(probs, key=probs.get).lower() == gt.lower())

def mutual_information(probs, gt, eps=1e-12):
    return -np.log2(max(probs.get(gt, eps), eps))

def get_vote_probs_gpt(messages, model):
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    text = response.choices[0].message.content.lower()

    probs = {c: 0.0 for c in CANDIDATES}
    for c in CANDIDATES:
        if c.lower() in text:
            probs[c] = 1.0

    if sum(probs.values()) == 0:
        probs = {c: 0.5 for c in CANDIDATES}

    Z = sum(probs.values())
    return {k: v / Z for k, v in probs.items()}

# ==========================================
# Inference loop
# ==========================================
for _, row in tqdm(df_input.iterrows(), total=len(df_input)):
    raw_idx = row["raw_idx"]
    if raw_idx in processed:
        continue

    messages = row["messages"]
    gt = extract_ground_truth(messages)
    if gt is None or gt.lower() not in CANDIDATES_NORM:
        continue

    clean_msgs = strip_assistant_messages(messages)

    try:
        probs = get_vote_probs_gpt(clean_msgs, args.model_name)
    except Exception as e:
        print(f"API error at idx {raw_idx}: {e}")
        time.sleep(5)
        continue

    acc = accuracy_from_probs(probs, gt)
    mi = mutual_information(probs, gt)
    pred = max(probs, key=probs.get)

    rows.append({
        "raw_idx": raw_idx,
        "probs": probs,
        "predicted_vote": pred,
        "ground_truth": gt,
        "accuracy": acc,
        "mutual_inf": mi,
        "model_name": args.model_name
    })

    processed.add(raw_idx)

    # 🔹 checkpoint
    if len(rows) % args.save_every == 0:
        pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
        print(f"Checkpoint saved: {len(rows)} samples")

    time.sleep(args.sleep)

# ==========================================
# Final dataframe
# ==========================================
df_results = pd.DataFrame(rows)
df_final = df_primary.merge(df_results, on="raw_idx", how="left")

# ==========================================
# Vote correspondence metrics
# ==========================================
anes = df_final["ground_truth"].dropna().map(vote_to_numeric).to_numpy()
gpt = df_final["predicted_vote"].dropna().map(vote_to_numeric).to_numpy()

metrics = {}
metrics["proportion_agreement"] = np.mean(anes == gpt)
metrics["cohen_kappa"] = cohen_kappa_score(anes, gpt)

if icc_available:
    df_tmp = pd.DataFrame({"anes": anes, "gpt": gpt})
    df_long = df_tmp.reset_index().melt(id_vars="index",
                                        value_vars=["anes", "gpt"],
                                        var_name="rater",
                                        value_name="vote")
    icc_df = pg.intraclass_corr(
        data=df_long,
        targets="index",
        raters="rater",
        ratings="vote"
    )
    metrics["ICC"] = icc_df.loc[icc_df["Type"] == "ICC2k", "ICC"].values[0]
else:
    metrics["ICC"] = None

for k, v in metrics.items():
    df_final[k] = v

# ==========================================
# Save outputs
# ==========================================
base = args.model_name.replace("/", "_")
df_final.to_csv(os.path.join(args.out_dir, f"{base}_final.csv"), index=False)
df_final.to_pickle(os.path.join(args.out_dir, f"{base}_final.pkl"))

print("\nSummary:")
print("Accuracy:", df_results["accuracy"].mean())
print("Mutual information:", df_results["mutual_inf"].mean())
print("Metrics:", metrics)
