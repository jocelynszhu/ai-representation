#%%
import pandas as pd
import numpy as np
from itertools import product
from itertools import combinations
from load_pairwise_data import load_pairwise_data
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt
#%%

def run_permutation_test(model_name):
    """
    Run permutation test for a given model and save results.
    
    Args:
        model_name: Name of the model to run the test for
        
    Returns:
        tuple: (all_diff_mean_flips, diff_mean_flips_original)
    """
    policies = pd.read_json("../self_selected_policies.jsonl", lines=True)
    prompts = ["prompt-0", "prompt-1", "prompt-2", "prompt-3", "prompt-4"]
    all_data = load_pairwise_data(model_name, prompts, policies_to_ignore=None, num_policies=20)
    
    # Calculate original test statistic
    mean_flips = all_data.groupby(["same_condition", "combined_prompt_name_1", "combined_prompt_name_2"]).flipped.mean()
    mean_flips = mean_flips.groupby("same_condition").mean()
    diff_mean_flips_original = mean_flips[False] - mean_flips[True]
    
    # Get all prompt combinations
    eight_prompts_one = set(all_data["combined_prompt_name_1"].unique())
    eight_prompts_two = set(all_data["combined_prompt_name_2"].unique())
    eight_prompts = list(eight_prompts_one | eight_prompts_two)
    eight_prompts_combinations = list(combinations(eight_prompts, 5))
    print(eight_prompts_combinations)
    # Run permutation test
    all_diff_mean_flips = []
    for chosen_prompts in tqdm(eight_prompts_combinations, desc=f"Running permutations for {model_name}"):
        modified_data = replace_roles_for_combination(all_data, chosen_prompts)
        mean_flips = modified_data.groupby(["same_condition", "combined_prompt_name_1", "combined_prompt_name_2"]).flipped.mean()
        mean_flips = mean_flips.groupby("same_condition").mean()
        diff_mean_flips = mean_flips[False] - mean_flips[True]
        all_diff_mean_flips.append(diff_mean_flips)
    
    return np.array(all_diff_mean_flips), diff_mean_flips_original

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

def save_results(all_diff_mean_flips, diff_mean_flips_original, model_name):
    """
    Save the numerical results.
    """
    # Create model-specific directory
    model_dir = f"../data/perm_test/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Calculate p-value
    p_value = sum(all_diff_mean_flips >= diff_mean_flips_original) / len(all_diff_mean_flips)
    # Save the data
    np.save(f"{model_dir}/all_diff_mean_flips.npy", all_diff_mean_flips)
    with open(f"{model_dir}/test_statistic.json", 'w') as f:
        json.dump({
            "diff_mean_flips_original": float(diff_mean_flips_original),
            "p_value": float(p_value)
        }, f)
    
    return p_value
import matplotlib.pyplot as plt
import numpy as np
import json

def plot_all_results(models, clean_name_mapping):
    """
    Create a single figure with three subplots in a row for all models.
    Reads data from saved files.

    Args:
        models: List of model names to plot
        clean_name_mapping: Dict mapping internal model names to clean display names
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # No shared axes
    fig.suptitle("Permutation Test Distribution", fontsize=16, y=0.9)

    for idx, (model_name, ax) in enumerate(zip(models, axes)):
        # Load the data
        model_dir = f"../data/perm_test/{model_name}"
        all_diff_mean_flips = np.load(f"{model_dir}/all_diff_mean_flips.npy")
        with open(f"{model_dir}/test_statistic.json", 'r') as f:
            stats = json.load(f)
            diff_mean_flips_original = stats["diff_mean_flips_original"]
            p_value = stats["p_value"]

        # Plot histogram without edge lines
        ax.hist(all_diff_mean_flips, bins=30, alpha=0.7, color='skyblue')

        # Plot test statistic line
        ax.axvline(x=diff_mean_flips_original, color='red', linestyle='--', linewidth=3)

        # Model title
        model_display = clean_name_mapping[model_name]
        ax.set_title(f'{model_display}', fontsize=12)

        # p-value annotation inside plot (top right corner)
        ax.text(0.8, 0.95, f'P-Value: {p_value:.3f}', ha='right', va='top',
                transform=ax.transAxes, fontsize=10)

        # Clean styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)

        # Remove duplicate axis labels
        if idx != 0:
            ax.set_ylabel('')
        else:
            ax.set_ylabel('Frequency')

        ax.set_xlabel('')

    # Shared x-axis label
    fig.text(0.5, 0.04, 'Test Statistic: Different - Same Condition', ha='center')

    # Single legend
    fig.legend(['Test Statistic Value'], loc='lower right', fontsize=12)

    plt.tight_layout(rect=[0.03, 0.06, 1, 0.93])
    plt.savefig("../data/perm_test/combined_permutation_test_plot.png", dpi=300, bbox_inches='tight')
    plt.close()



#%%

models = ["llama-3.2", "gpt-4o", "claude-3-sonnet-v2", ]
prompts = ["prompt-0", "prompt-1", "prompt-2", "prompt-3", "prompt-4"]
policies_to_ignore = None
# all_cross_data = []
# for model in models:
#     data = load_pairwise_data(model, prompts, policies_to_ignore=policies_to_ignore)
#     cross_data = data[data["same_condition"] == False].assign(model=model)
#     all_cross_data.append(cross_data)
# all_cross_data = pd.concat(all_cross_data)
# #%%
# by_model_pair = all_cross_data.groupby(["model", "combined_prompt_name_1", "combined_prompt_name_2"]).flipped.mean()\
#     .reset_index()
# # %%
# by_model_pair.groupby("model").flipped.mean()
# # %%

# # Run permutation test for each model

# for model in models:
#     print(f"\nRunning permutation test for {model}")
#     all_diff_mean_flips, diff_mean_flips_original = run_permutation_test(model)
#     p_value = save_results(all_diff_mean_flips, diff_mean_flips_original, model)
#     print(f"P-value for {model}: {p_value}")
#%%
# Create combined plot from saved data
clean_name_mapping = json.load(open("../clean_name_mapping.json"))
plot_all_results(models, clean_name_mapping)

# %%
