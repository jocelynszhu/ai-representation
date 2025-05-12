#%%
import pandas as pd
import numpy as np
from itertools import product
#%%

policies = pd.read_json("../self_selected_policies.jsonl", lines=True)
#%%
policies.head()
#%%
policies.shape
#%%
#base_llm = "gpt-4o"
base_llm = "claude-3-sonnet"
prompts = ["prompt-1", "prompt-2", "prompt-3", "prompt-4"]
prompt_types = ["delegate", "trustee"]
all_prompts = [(prompt, prompt_type) for prompt, prompt_type in product(prompts, prompt_types)]
# %%
from itertools import combinations
combinations_prompts = list(combinations(all_prompts, 2))
# %%
combinations_prompts
#%%
def get_pairs(prompt1, prompt2, policies_to_ignore=None):
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
        print(f"Removed {num_removed} rows with NA votes")
        merged['flipped'] = merged['vote_1'] != merged['vote_2']
        all_data.append(merged)
    all_data_df = pd.concat(all_data)
    all_data_df["same_condition"] = all_data_df["source_1"] == all_data_df["source_2"]
    all_data_df["combined_prompt_name_1"] = all_data_df["prompt1"] + "_" + all_data_df["source_1"]
    all_data_df["combined_prompt_name_2"] = all_data_df["prompt2"] + "_" + all_data_df["source_2"]
    return all_data_df
#%%
all_data = []
for prompt1, prompt2 in combinations_prompts:
    print(prompt1, prompt2)
    data = get_pairs(prompt1, prompt2)
    all_data.append(data)
# %%
all_data = pd.concat(all_data)
#%%
all_data.head()
# %%
mean_flips = all_data.groupby(["same_condition"]).flipped.mean()
diff_mean_flips_original = mean_flips[False] - mean_flips[True]
diff_mean_flips_original
#%%
eight_prompts_one = set(all_data["combined_prompt_name_1"].unique())
eight_prompts_two = set(all_data["combined_prompt_name_2"].unique())
eight_prompts = list(eight_prompts_one | eight_prompts_two)
eight_prompts_combinations = list(combinations(eight_prompts, 4))
#%%
# %%
def replace_roles_for_combination(data, chosen_prompts):
    """
    For a given combination of 4 prompts, replace the roles:
    - For the chosen prompts: set role to delegate
    - For all other prompts: set role to trustee
    
    Args:
        data: DataFrame containing all the vote comparison data
        chosen_prompts: List of 4 (prompt_name, prompt_type) tuples to set as delegate
    """
    # Create a copy to avoid modifying original
    modified_data = data.copy()
    
    
    # Replace source_1 roles
    modified_data['source_1'] = modified_data.apply(
        lambda row: 'trustee' if row['combined_prompt_name_1'] in chosen_prompts else 'delegate', 
        axis=1
    )
    
    # Replace source_2 roles
    modified_data['source_2'] = modified_data.apply(
        lambda row: 'trustee' if row['combined_prompt_name_2'] in chosen_prompts else 'delegate',
        axis=1
    )
    
    # Update same_condition flag
    modified_data['same_condition'] = modified_data['source_1'] == modified_data['source_2']
    
    return modified_data

# %%
all_diff_mean_flips = []
for i, chosen_prompts in enumerate(eight_prompts_combinations):
    print(f"Processing {i+1}/{len(eight_prompts_combinations)}")
    modified_data = replace_roles_for_combination(all_data, chosen_prompts)
    mean_flips = modified_data.groupby(["same_condition"]).flipped.mean()
    diff_mean_flips = mean_flips[False] - mean_flips[True]
    all_diff_mean_flips.append(diff_mean_flips)
# %%
all_diff_mean_flips = np.array(all_diff_mean_flips)
# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(all_diff_mean_flips, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(x=diff_mean_flips_original, color='red', linestyle='--', label='Original Difference')
plt.xlabel('Difference in Mean Flips (Different - Same Condition)')
plt.ylabel('Frequency')
plt.title('Distribution of Mean Flip Differences Across Prompt Combinations')
plt.legend()
plt.show()

# %%
chosen_prompts
# %%
modified_data
# %%
