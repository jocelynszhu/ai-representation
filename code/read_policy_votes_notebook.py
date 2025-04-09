# %% [markdown]
# # Policy Vote Analysis
# This notebook analyzes the voting data from two JSONL files containing policy votes.

# %%
import json
from pathlib import Path
from collections import Counter
import os
import pandas as pd

# %%
def read_jsonl_file(file_path):
    """Read a JSONL file and return a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def analyze_votes(data):
    """Analyze the voting data and return statistics."""
    vote_counts = Counter(entry['vote'] for entry in data)
    total_votes = len(data)
    
    return {
        'total_votes': total_votes,
        'yes_votes': vote_counts['yes'],
        'no_votes': vote_counts['no'],
        'yes_percentage': (vote_counts['yes'] / total_votes) * 100,
        'no_percentage': (vote_counts['no'] / total_votes) * 100
    }

# %%
# Define file paths
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
data_dir = current_dir.parent / 'data'
d_policy_file = data_dir / 'd_policy_1_votes.jsonl'
t_policy_file = data_dir / 't_policy_1_votes.jsonl'

print(f"Looking for files in: {data_dir}")
print(f"d_policy_file exists: {d_policy_file.exists()}")
print(f"t_policy_file exists: {t_policy_file.exists()}")

# %%
# Read the files
d_policy_data = read_jsonl_file(d_policy_file)
t_policy_data = read_jsonl_file(t_policy_file)

print(f"Read {len(d_policy_data)} entries from d_policy_1_votes.jsonl")
print(f"Read {len(t_policy_data)} entries from t_policy_1_votes.jsonl")

# %%
# Analyze d_policy_1_votes.jsonl
d_analysis = analyze_votes(d_policy_data)
print("\nAnalysis of d_policy_1_votes.jsonl:")
print(f"Total votes: {d_analysis['total_votes']}")
print(f"Yes votes: {d_analysis['yes_votes']} ({d_analysis['yes_percentage']:.2f}%)")
print(f"No votes: {d_analysis['no_votes']} ({d_analysis['no_percentage']:.2f}%)")

# %%
# Analyze t_policy_1_votes.jsonl
t_analysis = analyze_votes(t_policy_data)
print("\nAnalysis of t_policy_1_votes.jsonl:")
print(f"Total votes: {t_analysis['total_votes']}")
print(f"Yes votes: {t_analysis['yes_votes']} ({t_analysis['yes_percentage']:.2f}%)")
print(f"No votes: {t_analysis['no_votes']} ({t_analysis['no_percentage']:.2f}%)")

# %%
# Example: Look at the first few entries from each file
print("\nFirst 3 entries from d_policy_1_votes.jsonl:")
for i, entry in enumerate(d_policy_data[:3]):
    print(f"\nEntry {i+1}:")
    print(f"Vote: {entry['vote']}")
    print(f"Reasoning: {entry['reasoning'][:200]}...")  # Show first 200 chars of reasoning

# %%
# Example: Look at the first few entries from t_policy_1_votes.jsonl
print("\nFirst 3 entries from t_policy_1_votes.jsonl:")
for i, entry in enumerate(t_policy_data[:3]):
    print(f"\nEntry {i+1}:")
    print(f"Vote: {entry['vote']}")
    print(f"Reasoning: {entry['reasoning'][:200]}...")  # Show first 200 chars of reasoning

# %%
# Create a combined DataFrame
# First, create DataFrames for each file
d_df = pd.DataFrame(d_policy_data)
t_df = pd.DataFrame(t_policy_data)

# Rename columns to distinguish between d and t
d_df = d_df.rename(columns={
    'reasoning': 'd_reason',
    'vote': 'd_vote'
})

t_df = t_df.rename(columns={
    'reasoning': 't_reason',
    'vote': 't_vote'
})

# Combine the DataFrames
combined_df = pd.concat([d_df, t_df], axis=1)

# Display the first few rows
print("\nCombined DataFrame (first 5 rows):")
print(combined_df.head())

# Filter to show only rows where d_vote and t_vote differ
disagreements_df = combined_df[combined_df['d_vote'] != combined_df['t_vote']]

print("\nRows where d_vote and t_vote disagree:")
print(f"\nTotal disagreements: {len(disagreements_df)}")
print("\nSample of disagreements:")
for idx, row in disagreements_df.head().iterrows():
    print(f"\nIndex {idx}")
    print(f"d_vote: {row['d_vote']}")
    print(f"d reasoning: {row['d_reason'][:200]}...")
    print(f"t_vote: {row['t_vote']}")
    print(f"t reasoning: {row['t_reason'][:200]}...")

# %%
# Save disagreements to CSV
output_path = '../data/vote_disagreements.csv'
disagreements_df.to_csv(output_path, index=True)
print(f"\nDisagreements saved to {output_path}")

# %%
