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
    
    # Calculate average variance across GPT-4o and Claude for sorting
    avg_variance = (model_stats["GPT-4o"][('mean', 'delegate')] + model_stats["GPT-4o"][('mean', 'trustee')] +
                   model_stats["Claude"][('mean', 'delegate')] + model_stats["Claude"][('mean', 'trustee')]) / 4
    
    # Sort policies by average variance
    sorted_policies = avg_variance.sort_values(ascending=True).index
    
    # Apply the same sorting to all models
    for model_name in model_stats.keys():
        model_stats[model_name] = model_stats[model_name].reindex(sorted_policies)
    
    # Create the figure with subplots
    n_models = len(model_variance_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 11))
    if n_models == 1:
        axes = [axes]
    sns.set_style("white")  # Changed from "whitegrid" to "white" to remove gridlines
    
    # Add suptitle with adjusted position and font size
    plt.suptitle("Variance of Votes in Delegate vs. Trustee Conditions", y=0.98, fontsize=20)
    
    # Set the positions and width for the bars
    y_pos = np.arange(len(model_stats[list(model_stats.keys())[0]]))
    width = 0.25
    
    # Define colors using seaborn's colorblind-friendly palette
    delegate_color = '#2166ac'  # Darker blue for delegate
    trustee_color = '#fc8d59'   # Darker red for trustee
    
    # Plot data for each model
    for i, (ax, (model_name, stats)) in enumerate(zip(axes, model_stats.items())):
        # Plot bars
        ax.barh(y_pos - width/2, stats[('mean', 'delegate')], width, 
                label='Delegate', alpha=1, color=delegate_color)
        ax.barh(y_pos + width/2, stats[('mean', 'trustee')], width, 
                label='Trustee', alpha=1, color=trustee_color)
        
        # Add individual observations
        variance_df = model_variance_dict[model_name]
        for i, policy_id in enumerate(stats.index):
            policy_data = variance_df[variance_df['policy_id'] == policy_id]
            delegate_data = policy_data[policy_data['source'] == 'delegate']['var']
            trustee_data = policy_data[policy_data['source'] == 'trustee']['var']
            ax.scatter(delegate_data, [i - width/2] * len(delegate_data), 
                      color=delegate_color, alpha=0.4, s=30)
            ax.scatter(trustee_data, [i + width/2] * len(trustee_data), 
                      color=trustee_color, alpha=0.4, s=30)
        
        # Customize the plot
        ax.set_yticks(y_pos)
        if show_labels and model_name == list(model_stats.keys())[0]:
            ax.set_yticklabels([f'Policy {pid}' for pid in stats.index], fontsize=14)
        else:
            ax.set_yticklabels([])
        
        # Only show x-axis label on the middle plot
        if model_name == list(model_stats.keys())[1]:
            ax.set_xlabel('Variance', fontsize=16)
        else:
            ax.set_xlabel('')
            
        ax.set_title(model_name, fontsize=16)
        
        # Set tick label size
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Only add legend to the last plot with larger font size
        if model_name == list(model_stats.keys())[-1]:
            ax.legend(loc='lower right', fontsize=18)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('../data/plots/variance_comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    return model_stats

# Example usage:
prompts = ["prompt-0", "prompt-1", "prompt-2", "prompt-3", "prompt-4"]
models = {
    "Llama 3.2": "llama-3.2",
    "GPT-4o": "gpt-4o",
    "Claude": "claude-3-sonnet-v2",
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
    print(sum(list(stats[('mean', 'delegate')] > stats[('mean', 'trustee')].values)))
    print(f"{model_name}: {delegate_higher:.2%} of policies have higher delegate variance")


single = model_variance_dict["Llama 3.2"]
delegates = single[single["source"] == "delegate"]
trustees = single[single["source"] == "trustee"]
delegates.groupby(["prompt"]).agg({"var": "mean"}).reset_index()
#%%
