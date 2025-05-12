#%%
import pandas as pd
import numpy as np
from itertools import product
from itertools import combinations
from load_pairwise_data import load_pairwise_data
#%%

policies = pd.read_json("../self_selected_policies.jsonl", lines=True)
#%%
policies.head()
#%%
policies.shape
#%%
#base_llm = "gpt-4o"
base_llm = "claude-3-sonnet"

prompts = ["prompt-0", "prompt-1", "prompt-2", "prompt-3", "prompt-4"]
all_data = load_pairwise_data(base_llm, prompts, policies_to_ignore=None)
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
