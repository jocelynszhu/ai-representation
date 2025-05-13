#%%
import pandas as pd
import numpy as np
from itertools import product
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from load_pairwise_data import load_votes

def load_model_votes(model_name, prompts):
    """Load and process votes for a specific model."""
    votes = load_votes(model_name, prompts)
    votes = votes[(votes['vote'] == 'No') | (votes['vote'] == 'Yes')]
    votes['vote_binary'] = (votes['vote'] == 'Yes').astype(int)
    variance = votes.groupby(['prompt', 'source', 'policy_id'])['vote_binary'].agg(['mean', 'var']).round(3).reset_index()
    return votes, variance

def plot_variance_comparison(model_variance_dict, show_labels=True):
    """
    Create side-by-side horizontal bar plots comparing delegate and trustee variances for multiple models.
    
    Args:
        model_variance_dict: Dictionary mapping model names to their variance dataframes
        show_labels: Whether to show y-axis labels (only applies to first plot)
    """
    # Calculate mean variance for each policy and source for all models
    model_stats = {}
    for model_name, variance_df in model_variance_dict.items():
        stats = variance_df.groupby(['policy_id', 'source'])['var'].agg(['mean']).unstack(level=1)
        model_stats[model_name] = stats
    
    # Sort policies by total variance for first model
    first_model = list(model_stats.keys())[0]
    model_stats[first_model][('total', '')] = model_stats[first_model][('mean', 'delegate')] + model_stats[first_model][('mean', 'trustee')]
    model_stats[first_model] = model_stats[first_model].sort_values(('total', ''), ascending=True)
    model_stats[first_model] = model_stats[first_model].drop(('total', ''), axis=1)
    
    # Use same policy order for all other models
    for model_name in list(model_stats.keys())[1:]:
        model_stats[model_name] = model_stats[model_name].reindex(model_stats[first_model].index)
    
    # Create the figure with subplots
    n_models = len(model_variance_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
    if n_models == 1:
        axes = [axes]
    sns.set_style("whitegrid")
    
    # Set the positions and width for the bars
    y_pos = np.arange(len(model_stats[first_model]))
    width = 0.35
    
    # Plot data for each model
    for ax, (model_name, stats) in zip(axes, model_stats.items()):
        # Plot bars
        ax.barh(y_pos - width/2, stats[('mean', 'delegate')], width, 
                label='Delegate', alpha=0.6, color='blue')
        ax.barh(y_pos + width/2, stats[('mean', 'trustee')], width, 
                label='Trustee', alpha=0.6, color='orange')
        
        # Add individual observations
        variance_df = model_variance_dict[model_name]
        for i, policy_id in enumerate(stats.index):
            policy_data = variance_df[variance_df['policy_id'] == policy_id]
            delegate_data = policy_data[policy_data['source'] == 'delegate']['var']
            trustee_data = policy_data[policy_data['source'] == 'trustee']['var']
            ax.scatter(delegate_data, [i - width/2] * len(delegate_data), 
                      color='blue', alpha=0.3, s=30)
            ax.scatter(trustee_data, [i + width/2] * len(trustee_data), 
                      color='orange', alpha=0.3, s=30)
        
        # Customize the plot
        ax.set_yticks(y_pos)
        if show_labels and model_name == first_model:
            ax.set_yticklabels([f'Policy {pid}' for pid in stats.index])
        else:
            ax.set_yticklabels([])
        ax.set_xlabel('Variance')
        ax.set_title(f'{model_name} - Delegate vs Trustee Variance\nby Policy with Individual Observations')
        ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    # Save the plot
    plt.savefig('variance_comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    return model_stats

# Example usage:
prompts = ["prompt-1", "prompt-2", "prompt-3", "prompt-4"]
models = {
    "GPT-4": "gpt-4o",
    "Claude": "claude-3-sonnet-v2",
    "Llama": "llama-3.2"
}

# Load data for all models
model_variance_dict = {}
for display_name, model_name in models.items():
    _, variance = load_model_votes(model_name, prompts)
    model_variance_dict[display_name] = variance

# Create the plot
model_stats = plot_variance_comparison(model_variance_dict)

# Calculate statistics for each model
for model_name, stats in model_stats.items():
    delegate_higher = (stats[('mean', 'delegate')] > stats[('mean', 'trustee')]).sum() / len(stats)
    print(f"{model_name}: {delegate_higher:.2%} of policies have higher delegate variance")
#%%
single = model_variance_dict["Llama"]
delegates = single[single["source"] == "delegate"]
trustees = single[single["source"] == "trustee"]
#%%
# # Calculate percentage of policies where delegate variance is higher than trustee variance
# gpt_delegate_higher = (model_stats["GPT-4"][('mean', 'delegate')] > model_stats["GPT-4"][('mean', 'trustee')]).sum() / len(model_stats["GPT-4"])
# claude_delegate_higher = (model_stats["Claude"][('mean', 'delegate')] > model_stats["Claude"][('mean', 'trustee')]).sum() / len(model_stats["Claude"])

# print(f"GPT-4: {gpt_delegate_higher:.2%} of policies have higher delegate variance")
# print(f"Claude: {claude_delegate_higher:.2%} of policies have higher delegate variance")

#%%
by_group = model_variance_dict["GPT-4"].groupby(["vote"]).agg({'reason': 'count'}).reset_index()
#%%
# biographies = pd.read_json("rep_biographies.jsonl", lines=True)\
#     .rename(columns={"ID": "id"})

# #%%
# merged = model_variance_dict["GPT-4"].merge(biographies, on='id', how='left')
# merged["vote_binary"] = (merged["vote"] == "Yes").astype(int)
# merged.groupby(['prompt', 'source', 'Political Affiliation']).agg({'vote_binary': 'mean'}).reset_index()
# #%%
# #%%
# vote_variance = model_variance_dict["GPT-4"].groupby(['prompt', 'source', 'policy_id'])['vote_binary'].agg(['mean', 'var']).round(3)\
#     .reset_index()

# # %%
# def plot_variance_comparison(vote_variance):
#     """Create a horizontal bar plot comparing delegate and trustee variances for each policy with individual observations."""
#     # Calculate mean variance for each policy and source
#     stats = vote_variance.groupby(['policy_id', 'source'])['var'].agg(['mean']).unstack(level=1)
    
#     # Sort policies by total variance
#     stats[('total', '')] = stats[('mean', 'delegate')] + stats[('mean', 'trustee')]
#     stats = stats.sort_values(('total', ''), ascending=True)
#     stats = stats.drop(('total', ''), axis=1)
    
#     # Create the plot
#     plt.figure(figsize=(12, 8))
#     sns.set_style("whitegrid")
    
#     # Set the positions and width for the bars
#     y_pos = np.arange(len(stats))
#     width = 0.35
    
#     # Create the bars
#     plt.barh(y_pos - width/2, stats[('mean', 'delegate')], width, 
#              label='Delegate', alpha=0.6)
#     plt.barh(y_pos + width/2, stats[('mean', 'trustee')], width, 
#              label='Trustee', alpha=0.6)
    
#     # Add individual observations as dots
#     for i, policy_id in enumerate(stats.index):
#         # Get all observations for this policy
#         policy_data = vote_variance[vote_variance['policy_id'] == policy_id]
        
#         # Plot delegate observations
#         delegate_data = policy_data[policy_data['source'] == 'delegate']['var']
#         plt.scatter(delegate_data, [i - width/2] * len(delegate_data), 
#                    color='blue', alpha=0.3, s=30)
        
#         # Plot trustee observations
#         trustee_data = policy_data[policy_data['source'] == 'trustee']['var']
#         plt.scatter(trustee_data, [i + width/2] * len(trustee_data), 
#                    color='orange', alpha=0.3, s=30)
    
#     # Customize the plot
#     plt.yticks(y_pos, [f'Policy {pid}' for pid in stats.index])
#     plt.xlabel('Variance')
#     plt.title('Comparison of Delegate vs Trustee Variance by Policy\nwith Individual Observations')
#     plt.legend()
    
#     # Adjust layout
#     plt.tight_layout()
#     plt.show()
    
#     # Save the plot
#     plt.savefig('variance_comparison_plot.png', dpi=300, bbox_inches='tight')
#     plt.close()
#     return stats

# # Call the function to create the plot
# stats = plot_variance_comparison(vote_variance)

# # %%
# (stats[('mean', 'delegate')] > stats[('mean', 'trustee')]).sum() / len(stats)
# # %%
# stats
# # %%
# vote_variance[vote_variance['policy_id'] == 6]
# # %%

# # %%
