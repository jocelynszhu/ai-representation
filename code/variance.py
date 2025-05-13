

#%%
import pandas as pd
import numpy as np
from itertools import product
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from load_pairwise_data import load_votes
#%%
votes = load_votes("claude-3-sonnet-v2", ["prompt-1", "prompt-2", "prompt-3", "prompt-4"])
#%%
biographies = pd.read_json("rep_biographies.jsonl", lines=True)\
    .rename(columns={"ID": "id"})

#%%
merged = votes.merge(biographies, on='id', how='left')
merged["vote_binary"] = (merged["vote"] == "Yes").astype(int)
merged.groupby(['prompt', 'source', 'Political Affiliation']).agg({'vote_binary': 'mean'}).reset_index()
#%%
# Convert votes to binary (0 for No, 1 for Yes)
votes['vote_binary'] = (votes['vote'] == 'Yes').astype(int)

# Calculate variance grouped by prompt and source
vote_variance = votes.groupby(['prompt', 'source', 'policy_id'])['vote_binary'].agg(['mean', 'var']).round(3)\
    .reset_index()

# %%
def plot_variance_comparison():
    """Create a horizontal bar plot comparing delegate and trustee variances for each policy with 95% confidence intervals."""
    # Calculate mean variance and standard error for each policy and source
    stats = vote_variance.groupby(['policy_id', 'source'])['var'].agg(['mean', 'std', 'count']).unstack(level=1)
    
    # Calculate 95% confidence intervals for the mean
    # SE = std/sqrt(n)
    for source in ['delegate', 'trustee']:
        stats[('ci', source)] = 1.96 * (stats[('std', source)] / np.sqrt(stats[('count', source)]))
    
    # Sort policies by total variance
    stats[('total', '')] = stats[('mean', 'delegate')] + stats[('mean', 'trustee')]
    stats = stats.sort_values(('total', ''), ascending=True)
    stats = stats.drop(('total', ''), axis=1)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Set the positions and width for the bars
    y_pos = np.arange(len(stats))
    width = 0.35
    
    # Create the bars
    plt.barh(y_pos - width/2, stats[('mean', 'delegate')], width, 
             label='Delegate', alpha=0.6)
    plt.barh(y_pos + width/2, stats[('mean', 'trustee')], width, 
             label='Trustee', alpha=0.6)
    
    # Add error bars
    plt.errorbar(stats[('mean', 'delegate')], y_pos - width/2,
                xerr=stats[('ci', 'delegate')],
                fmt='none', color='black', capsize=5)
    plt.errorbar(stats[('mean', 'trustee')], y_pos + width/2,
                xerr=stats[('ci', 'trustee')],
                fmt='none', color='black', capsize=5)
    
    # Customize the plot
    plt.yticks(y_pos, [f'Policy {pid}' for pid in stats.index])
    plt.xlabel('Variance')
    plt.title('Comparison of Delegate vs Trustee Variance by Policy\nwith 95% Confidence Intervals')
    plt.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('variance_comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

# Call the function to create the plot
plot_variance_comparison()

# %%
