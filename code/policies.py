#%%
import pandas as pd
import numpy as np
from itertools import product
from load_pairwise_data import load_pairwise_data
import matplotlib.pyplot as plt
import seaborn as sns
# %%
#base_llm = "gpt-4o"
base_llm = "claude-3-sonnet-v2"
#$base_llm = "llama-3.2"
prompts = ["prompt-1", "prompt-2", "prompt-3", "prompt-4"]
#$rompts = [ "prompt-3", "prompt-4"]
policies_to_ignore = None
# Load data for both models
gpt_data = load_pairwise_data("gpt-4o", prompts, policies_to_ignore=policies_to_ignore)
claude_data = load_pairwise_data("claude-3-sonnet-v2", prompts, policies_to_ignore=policies_to_ignore)

# Add model identifier
gpt_data['model'] = 'GPT-4o'
claude_data['model'] = 'Claude'

# Combine the data
all_data = pd.concat([gpt_data, claude_data], ignore_index=True)
# %%
biographies = pd.read_json("rep_biographies.jsonl", lines=True)
biographies['id_1'] = biographies['ID']
# %%
joined = all_data.merge(biographies, on='id_1', how='left')
simple = joined.copy()
simple = simple.rename({"Political Affiliation": "political_affiliation"})

simple["flipped"] = simple["flipped"].astype(int)

simple = simple[simple.same_condition == False]
#simple = simple[simple.policy_id != 20]

# %%
demographics = joined.columns[17:]
for demographic in demographics:
    print(demographic)
    print(simple.groupby(demographic)["flipped"].value_counts(normalize=True).unstack())
    print("\n")
    print(simple.groupby(demographic)["flipped"].value_counts(normalize=False).unstack())
# %%
#%%
by_topic = simple.groupby("policy_id")["flipped"].value_counts(normalize=True).unstack()\
.sort_values(by=1, ascending=False)
policies = pd.read_json("../self_selected_policies.jsonl", lines=True)  
policies["policy_id"] = policies["id"]
merged_by_topic = by_topic.merge(policies, on="policy_id", how="left")
# %%
# Set display options to show all text in DataFrame without truncation
# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)

#erged_by_topic
# %%
grouped = simple.groupby(["policy_id", "model", "combined_prompt_name_1", "combined_prompt_name_2"])["flipped"].value_counts(normalize=True).unstack()
grouped.reset_index(inplace=True)

# %%

def plot_flip_rates(grouped_merged):
    """Create a horizontal bar chart showing flip rates for each policy comparing GPT-4o and Claude."""
    # Calculate mean flip rate and confidence intervals for each policy and model
    stats = grouped_merged.groupby(['policy_id', 'statement', 'model'])[1].agg(['mean', 'std', 'count']).reset_index()
    stats['ci'] = 1.645 * (stats['std'] / np.sqrt(stats['count']))  # 95% CI = mean ± 1.96 * SE

    # Calculate average flip rate across models for sorting
    avg_flip_rates = stats.groupby('policy_id')['mean'].mean().sort_values(ascending=True)
    policy_order = avg_flip_rates.index.tolist()
    
    # Create the plot
    plt.figure(figsize=(15, 10))
  #  sns.set_style("whitegrid")

    # Set up the positions for the bars
    n_policies = len(policy_order)
    x = np.arange(n_policies)
    width = 0.35
    color_mapping = {'GPT-4o': '#1f77b4', 'Claude': '#ff7f0e'}
    # Plot bars for each model
    for i, model in enumerate(['GPT-4o', 'Claude']):
        model_data = stats[stats['model'] == model]
        model_data = model_data.set_index('policy_id').reindex(policy_order)
        
        plt.barh(x + i*width, model_data['mean'].values, width,
                label=model, alpha=1, color=color_mapping[model])
        
        # Add error bars
        plt.errorbar(model_data['mean'].values, x + i*width,
                    xerr=model_data['ci'].values,
                    fmt='none', color='grey', capsize=5, alpha=1)

    # Customize the plot
    plt.yticks(x + width/2, [stats[stats['policy_id']==pid]['statement'].iloc[0] for pid in policy_order])
    plt.xlabel('Proportion of Vote Changes')
    plt.title('Vote Change Proportion by Policy', fontsize=16)
    plt.legend(loc='lower right', fontsize=16)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.show()
    plt.savefig('../data/plots/flip_rates_comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    return stats

# Call the function to create the plot
policies = pd.read_json("../self_selected_policies.jsonl", lines=True)  
policies["policy_id"] = policies["id"]
grouped_merged = grouped.merge(policies, on="policy_id", how="left")
#%%
stats = plot_flip_rates(grouped_merged)

# %%
# Assume 'stats' is your DataFrame as shown above

# Create a new column with the formatted string
stats['mean_ci'] = stats.apply(lambda row: f"{row['mean']:.3f} ± {row['ci']:.3f}", axis=1)
mean_across_models = stats.groupby('statement')['mean'].mean().sort_values(ascending=False)
# Pivot the table
pivoted = stats.pivot(index=[ 'statement'], columns='model', values='mean_ci').reset_index()

# Optional: reorder columns if needed
pivoted = pivoted[['statement', 'Claude', 'GPT-4o']]
# Sort the pivoted table based on mean_across_models
pivoted = pivoted.set_index('statement').loc[mean_across_models.index].reset_index()

# Display the result
pivoted
# %
# %%

# %%
# Create LaTeX table with bold formatting for top 3 policies with highest flip rates per model
def create_latex_table(pivoted_df):
    # Create a copy of the dataframe for formatting
    formatted_df = pivoted_df.copy()
    
    # Format numbers to 2 decimal places and identify top 3 for each model
    for col in ['Claude', 'GPT-4o']:
        # Extract main values and format to 2 decimal places
        formatted_df[f'{col}_val'] = formatted_df[col].str.extract(r'(\d+\.\d+)').astype(float)
        formatted_df[col] = formatted_df[col].apply(lambda x: f"{float(x.split('±')[0]):.2f} ± {float(x.split('±')[1]):.2f}")
        
        # Get top 3 policies for this model
        top_3_policies = formatted_df.nlargest(3, f'{col}_val')['statement'].tolist()
        
        # Add bold formatting for top 3 policies for this model
        for policy in top_3_policies:
            mask = formatted_df['statement'] == policy
            formatted_df.loc[mask, col] = '\\textbf{' + formatted_df.loc[mask, col] + '}'
    
    # Generate LaTeX table
    latex_table = formatted_df.to_latex(
        index=False,
        escape=False,
        caption='Flip rates for different policies across models',
        label='tab:flip_rates',
        columns=['statement', 'Claude', 'GPT-4o']  # Only include needed columns
    )
    
    return latex_table

# Generate and print the LaTeX table
latex_output = create_latex_table(pivoted)
print(latex_output)

# Save to file
with open('../data/latex/flip_rates_table.tex', 'w') as f:
    f.write(latex_output)

# %%
