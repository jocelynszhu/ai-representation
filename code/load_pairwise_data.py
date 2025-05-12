#%%
import pandas as pd
import numpy as np
from itertools import product
from itertools import combinations
# %%
# %%
def get_pairs(prompt1, prompt2, base_llm, policies_to_ignore=None):
    """Compute the percentage of flipped votes between two trials.
    
    Args:
        prompt1: First prompt (e.g., ('prompt-1', 'delegate'))
        prompt2: Second prompt (e.g., ('prompt-2', 'delegate'))
    """

    prompt1_name = prompt1[0]
    prompt1_type = prompt1[1]
    prompt2_name = prompt2[0]
    prompt2_type = prompt2[1]

    all_data = []
    for i in range(1, 21):
        if policies_to_ignore is not None and i in policies_to_ignore:
            print(f"Skipping policy {i} because it is in the ignore list")
            continue
       # print(f"Processing policy {i}...")
        # Load votes for both trials
        try:
            votes1 = pd.read_json(f"../data/{prompt1_type}/{base_llm}/{prompt1_name}/{prompt1_type[0]}_policy_{i}_votes.jsonl", encoding='cp1252', lines=True)
        except:
            print(f"File not found: {prompt1_type}/{base_llm}/{prompt1_name}/{prompt1_type[0]}_policy_{i}_votes.jsonl")
            raise Exception(f"File not found: {prompt1_type}/{base_llm}/{prompt1_name}/{prompt1_type[0]}_policy_{i}_votes.jsonl")
          #  continue
        try:
            votes2 = pd.read_json(f"../data/{prompt2_type}/{base_llm}/{prompt2_name}/{prompt2_type[0]}_policy_{i}_votes.jsonl", encoding='cp1252', lines=True)
        except:
            print(f"File not found: {prompt2_type}/{base_llm}/{prompt2_name}/{prompt2_type[0]}_policy_{i}_votes.jsonl")
            raise Exception(f"File not found: {prompt2_type}/{base_llm}/{prompt2_name}/{prompt2_type[0]}_policy_{i}_votes.jsonl")
        
        # Add source and index columns
        votes1['source'] = prompt1_type
        votes2['source'] = prompt2_type
        votes1['idx'] = range(len(votes1))
        votes2['idx'] = range(len(votes2))
        
        # Merge on index
        merged = pd.merge(votes1, votes2, how='inner', on='idx', suffixes=('_1', '_2'))\
            .assign(policy_id=i, prompt1=prompt1_name, prompt2=prompt2_name)
        # Remove rows with NA votes from both sides
        merged = merged[
            (~merged['vote_1'].isin(['NA', 'na', 'N/A', 'n/a'])) & 
            (~merged['vote_2'].isin(['NA', 'na', 'N/A', 'n/a']))
        ]
        # Print number of rows removed due to NA votes
        num_removed = len(votes1) - len(merged)
      #  print(f"Removed {num_removed} rows with NA votes")
        merged['flipped'] = merged['vote_1'] != merged['vote_2']
        all_data.append(merged)
    all_data_df = pd.concat(all_data)
    all_data_df["same_condition"] = all_data_df["source_1"] == all_data_df["source_2"]
    all_data_df["combined_prompt_name_1"] = all_data_df["prompt1"] + "_" + all_data_df["source_1"]
    all_data_df["combined_prompt_name_2"] = all_data_df["prompt2"] + "_" + all_data_df["source_2"]
    return all_data_df
#%%
def load_pairwise_data(base_llm, prompts, policies_to_ignore=None):
    prompt_types = ["delegate", "trustee"]
    all_prompts = [(prompt, prompt_type) for prompt, prompt_type in product(prompts, prompt_types)]
    combinations_prompts = list(combinations(all_prompts, 2))
    all_data = []
    for prompt1, prompt2 in combinations_prompts:
        #print(prompt1, prompt2)
        data = get_pairs(prompt1, prompt2, base_llm,  policies_to_ignore=policies_to_ignore)
        all_data.append(data)
    all_data = pd.concat(all_data)
    return all_data
# %%
base_llm = "gpt-4o"
#base_llm = "claude-3-sonnet"

prompts = ["prompt-1", "prompt-2", "prompt-3", "prompt-4"]
all_data = load_pairwise_data(base_llm, prompts, policies_to_ignore=None)
# %%
# Set display options to show all text in DataFrame without truncation
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

all_data[(all_data["policy_id"] == 5) & (all_data["same_condition"] == False) & (all_data["flipped"] == True)]\
    .filter(["prompt1", "prompt2", "source_1", "source_2", "vote_1", "vote_2", "reason_1", "reason_2"])

# %%
