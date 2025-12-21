import pandas as pd
import json
from pathlib import Path
# ==========================================
# Paths
# ==========================================
# Define base directories
BASE_DIR ='/Users/zeynab/Desktop/opinion_polling/'

DATA_DIR = BASE_DIR+"data/"
OUTPUT_DIR = BASE_DIR+"result/"

# Ensure output directory exists
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input dataset path
file_path = DATA_DIR +"/anes_timeseries_2020_csv_20220210.csv"

# Output files
jsonl_filename = OUTPUT_DIR+ "anes_2020_finetune.jsonl"
json_filename = OUTPUT_DIR+ "anes_2020_chat_finetune.json"
csv_filename = OUTPUT_DIR+ "anes_2020_chat_finetune.csv"

# ==========================================
# 1. Load and Clean Data
# ==========================================
df = pd.read_csv(file_path)

# Map ANES 2020 Variables
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

df_study = df[cols_map.keys()].rename(columns=cols_map)

# Filter: Remove missing data and keep only Biden/Trump voters
demographic_cols = [c for c in df_study.columns if c != 'vote_choice']
df_clean = df_study[(df_study[demographic_cols] >= 0).all(axis=1)].copy()
df_clean = df_clean[df_clean['vote_choice'].isin([1, 2])]
print(df_study.shape)
print(df_clean.shape)
# ==========================================
# 2. Define Mappings (Text Generation)
# ==========================================
race_map = {1: "white", 2: "black", 3: "hispanic", 4: "asian", 5: "native American", 6: "mixed race"}
gender_map = {1: "man", 2: "woman"}
ideology_map = {
    1: "extremely liberal", 2: "liberal", 3: "slightly liberal", 4: "moderate",
    5: "slightly conservative", 6: "conservative", 7: "extremely conservative"
}
party_map = {
    1: "a strong democrat", 2: "a weak Democrat", 3: "an independent who leans Democratic",
    4: "an independent", 5: "an independent who leans Republican", 
    6: "a weak Republican", 7: "a strong Republican"
}
interest_map = {1: "very", 2: "somewhat", 3: "not very", 4: "not at all"}
discuss_map = {
    1: "I like to discuss politics with my family and friends.",
    2: "I never discuss politics with my family or friends."
}

def get_church_text(code):
    return "attend church" if code <= 2 else "do not attend church"

# ==========================================
# 3. Create Data Structures
# ==========================================
chat_objects = []
flat_rows = []

for _, row in df_clean.iterrows():
    try:
        # User Content (Backstory)
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
        
        # Assistant Content (Label)
        assistant_content = "Joe Biden" if row['vote_choice'] == 1 else "Donald Trump"
        
        # JSON/JSONL structure
        chat_entry = {
            "messages": [
                {"role": "system", "content": "You are an expert political analyst specializing in US elections and voting behavior. Your task is to analyze the demographic profile provided in the text and predict the vote choice in the 2020 Election. Output strictly one name: 'Donald Trump' or 'Joe Biden'."},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        }
        chat_objects.append(chat_entry)
        
        # Flattened CSV structure
        flat_rows.append({
            "user_content": user_content,
            "assistant_content": assistant_content
        })
        
    except KeyError:
        continue

# ==========================================
# 4. Save Outputs
# ==========================================
# JSONL
with open(jsonl_filename, 'w') as f:
    for entry in chat_objects:
        json.dump(entry, f)
        f.write('\n')
print(f"Saved OpenAI JSONL to: {jsonl_filename}")

# JSON
with open(json_filename, 'w') as f:
    json.dump(chat_objects, f, indent=4)
print(f"Saved Standard JSON to: {json_filename}")

# CSV
df_export = pd.DataFrame(flat_rows)
df_export.to_csv(csv_filename, index=False)
print(f"Saved CSV to: {csv_filename}")

# Preview CSV
print("\n--- CSV Data Preview ---")
print(df_export.head(2))
