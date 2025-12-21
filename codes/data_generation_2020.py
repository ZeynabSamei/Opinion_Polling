import pandas as pd
import json
from pathlib import Path

# ==========================================
# Paths (VS Code & Git friendly)
# ==========================================

# Directory where THIS script lives
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "result"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input dataset
file_path = DATA_DIR / "anes_timeseries_2020_csv_20220210.csv"

# Output files
jsonl_filename = OUTPUT_DIR / "anes_2020_finetune.jsonl"
json_filename  = OUTPUT_DIR / "anes_2020_chat_finetune.json"
csv_filename   = OUTPUT_DIR / "anes_2020_chat_finetune.csv"

# ==========================================
# 1. Load and Clean Data
# ==========================================

df = pd.read_csv(file_path)

cols_map = {
    'V202073': 'vote_choice',
    'V201600': 'gender',
    'V201549x': 'race',
    'V201507x': 'age',
    'V201200': 'ideology',
    'V201231x': 'party_id',
    'V201006': 'pol_interest',
    'V201453': 'church_attendance',
    'V202022': 'discuss_politics'
}

df_study = df[list(cols_map.keys())].rename(columns=cols_map)

demographic_cols = [c for c in df_study.columns if c != 'vote_choice']
df_clean = df_study[(df_study[demographic_cols] >= 0).all(axis=1)].copy()
df_clean = df_clean[df_clean['vote_choice'].isin([1, 2])]

print("Raw shape:", df_study.shape)
print("Clean shape:", df_clean.shape)

# ==========================================
# 2. Text Mappings
# ==========================================

race_map = {
    1: "white", 2: "black", 3: "hispanic",
    4: "asian", 5: "native American", 6: "mixed race"
}
gender_map = {1: "man", 2: "woman"}
ideology_map = {
    1: "extremely liberal", 2: "liberal", 3: "slightly liberal",
    4: "moderate", 5: "slightly conservative",
    6: "conservative", 7: "extremely conservative"
}
party_map = {
    1: "a strong Democrat",
    2: "a weak Democrat",
    3: "an independent who leans Democratic",
    4: "an independent",
    5: "an independent who leans Republican",
    6: "a weak Republican",
    7: "a strong Republican"
}
interest_map = {
    1: "very", 2: "somewhat",
    3: "not very", 4: "not at all"
}
discuss_map = {
    1: "I like to discuss politics with my family and friends.",
    2: "I never discuss politics with my family or friends."
}

def get_church_text(code):
    return "attend church" if code <= 2 else "do not attend church"

# ==========================================
# 3. Build Chat Dataset
# ==========================================

chat_objects = []
flat_rows = []

SYSTEM_PROMPT = (
    "You are an expert political analyst specializing in US elections and voting behavior. "
    "Your task is to analyze the demographic profile provided in the text and predict the vote "
    "choice in the 2020 Election. Output strictly one name: 'Donald Trump' or 'Joe Biden'."
)

for _, row in df_clean.iterrows():
    try:
        user_content = (
            f"Racially, I am {race_map[row['race']]}. "
            f"I am a {gender_map[row['gender']]}. "
            f"I am {int(row['age'])} years old. "
            f"Ideologically, I am {ideology_map[row['ideology']]}. "
            f"Politically, I am {party_map[row['party_id']]}. "
            f"I {get_church_text(row['church_attendance'])}. "
            f"I am {interest_map[row['pol_interest']]} interested in politics. "
            f"{discuss_map[row['discuss_politics']]} "
            "In the 2020 presidential election, I voted for "
        )

        assistant_content = (
            "Joe Biden" if row['vote_choice'] == 1 else "Donald Trump"
        )

        chat_objects.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        })

        flat_rows.append({
            "user_content": user_content,
            "assistant_content": assistant_content
        })

    except KeyError:
        continue

# ==========================================
# 4. Save Outputs
# ==========================================

with open(jsonl_filename, "w") as f:
    for entry in chat_objects:
        json.dump(entry, f)
        f.write("\n")

with open(json_filename, "w") as f:
    json.dump(chat_objects, f, indent=2)

pd.DataFrame(flat_rows).to_csv(csv_filename, index=False)

print("Saved:")
print(" -", jsonl_filename)
print(" -", json_filename)
print(" -", csv_filename)
