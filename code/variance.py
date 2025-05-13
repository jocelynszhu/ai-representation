#%%
import pandas as pd
import numpy as np
from itertools import product
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from load_pairwise_data import load_votes
#%%
# Load votes for both models
gpt_votes = load_votes("gpt-4o", ["prompt-1", "prompt-2", "prompt-3", "prompt-4"])
claude_votes = load_votes("claude-3-sonnet-v2", ["prompt-1", "prompt-2", "prompt-3", "prompt-4"])

# Filter for valid votes
gpt_votes = gpt_votes[(gpt_votes['vote'] == 'No') | (gpt_votes['vote'] == 'Yes')]
claude_votes = claude_votes[(claude_votes['vote'] == 'No') | (claude_votes['vote'] == 'Yes')]

# Convert votes to binary (0 for No, 1 for Yes)
gpt_votes['vote_binary'] = (gpt_votes['vote'] == 'Yes').astype(int)
claude_votes['vote_binary'] = (claude_votes['vote'] == 'Yes').astype(int)

# Calculate variance grouped by prompt and source for both models
gpt_variance = gpt_votes.groupby(['prompt', 'source', 'policy_id'])['vote_binary'].agg(['mean', 'var']).round(3).reset_index()
claude_variance = claude_votes.groupby(['prompt', 'source', 'policy_id'])['vote_binary'].agg(['mean', 'var']).round(3).reset_index()

#%%
def plot_variance_comparison(gpt_variance, claude_variance):
    """Create side-by-side horizontal bar plots comparing delegate and trustee variances for GPT-4 and Claude."""
    # Calculate mean variance for each policy and source for both models
    gpt_stats = gpt_variance.groupby(['policy_id', 'source'])['var'].agg(['mean']).unstack(level=1)
    claude_stats = claude_variance.groupby(['policy_id', 'source'])['var'].agg(['mean']).unstack(level=1)
    
    # Sort policies by total variance for GPT-4
    gpt_stats[('total', '')] = gpt_stats[('mean', 'delegate')] + gpt_stats[('mean', 'trustee')]
    gpt_stats = gpt_stats.sort_values(('total', ''), ascending=True)
    gpt_stats = gpt_stats.drop(('total', ''), axis=1)
    
    # Use same policy order for Claude
    claude_stats = claude_stats.reindex(gpt_stats.index)
    
    # Create the figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    sns.set_style("whitegrid")
    
    # Set the positions and width for the bars
    y_pos = np.arange(len(gpt_stats))
    width = 0.35
    
    # Plot GPT-4 data
    ax1.barh(y_pos - width/2, gpt_stats[('mean', 'delegate')], width, color='blue', label='Delegate', alpha=0.6)
    ax1.barh(y_pos + width/2, gpt_stats[('mean', 'trustee')], width, 
             label='Trustee', alpha=0.6, color='orange')
    
    # Add individual observations for GPT-4
    for i, policy_id in enumerate(gpt_stats.index):
        policy_data = gpt_variance[gpt_variance['policy_id'] == policy_id]
        delegate_data = policy_data[policy_data['source'] == 'delegate']['var']
        trustee_data = policy_data[policy_data['source'] == 'trustee']['var']
        ax1.scatter(delegate_data, [i - width/2] * len(delegate_data), 
                   color='blue', alpha=0.3, s=30)
        ax1.scatter(trustee_data, [i + width/2] * len(trustee_data), 
                   color='orange', alpha=0.3, s=30)
    
    # Plot Claude data
    ax2.barh(y_pos - width/2, claude_stats[('mean', 'delegate')], width, 
             label='Delegate', alpha=0.6, color='blue')
    ax2.barh(y_pos + width/2, claude_stats[('mean', 'trustee')], width, 
             label='Trustee', alpha=0.6, color='orange')
    
    # Add individual observations for Claude
    for i, policy_id in enumerate(claude_stats.index):
        policy_data = claude_variance[claude_variance['policy_id'] == policy_id]
        delegate_data = policy_data[policy_data['source'] == 'delegate']['var']
        trustee_data = policy_data[policy_data['source'] == 'trustee']['var']
        ax2.scatter(delegate_data, [i - width/2] * len(delegate_data), 
                   color='blue', alpha=0.3, s=30)
        ax2.scatter(trustee_data, [i + width/2] * len(trustee_data), 
                   color='orange', alpha=0.3, s=30)
    
    # Customize the plots
    for ax, title, show_labels in zip([ax1, ax2], ['GPT-4', 'Claude'], [True, False]):
        ax.set_yticks(y_pos)
        if show_labels:
            ax.set_yticklabels([f'Policy {pid}' for pid in gpt_stats.index])
        else:
            ax.set_yticklabels([])
        ax.set_xlabel('Variance')
        ax.set_title(f'{title} - Delegate vs Trustee Variance by Policy\nwith Individual Observations')
        ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    # Save the plot
    plt.savefig('variance_comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    return gpt_stats, claude_stats

# Call the function to create the plots
gpt_stats, claude_stats = plot_variance_comparison(gpt_variance, claude_variance)

#%%
# Calculate percentage of policies where delegate variance is higher than trustee variance
gpt_delegate_higher = (gpt_stats[('mean', 'delegate')] > gpt_stats[('mean', 'trustee')]).sum() / len(gpt_stats)
claude_delegate_higher = (claude_stats[('mean', 'delegate')] > claude_stats[('mean', 'trustee')]).sum() / len(claude_stats)

print(f"GPT-4: {gpt_delegate_higher:.2%} of policies have higher delegate variance")
print(f"Claude: {claude_delegate_higher:.2%} of policies have higher delegate variance")

#%%
by_group = gpt_votes.groupby(["vote"]).agg({'reason': 'count'}).reset_index()
#%%
# biographies = pd.read_json("rep_biographies.jsonl", lines=True)\
#     .rename(columns={"ID": "id"})

# #%%
# merged = gpt_votes.merge(biographies, on='id', how='left')
# merged["vote_binary"] = (merged["vote"] == "Yes").astype(int)
# merged.groupby(['prompt', 'source', 'Political Affiliation']).agg({'vote_binary': 'mean'}).reset_index()
# #%%
# #%%
# vote_variance = gpt_votes.groupby(['prompt', 'source', 'policy_id'])['vote_binary'].agg(['mean', 'var']).round(3)\
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
