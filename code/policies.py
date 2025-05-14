#%%
import pandas as pd
import numpy as np
from itertools import product
from load_pairwise_data import load_pairwise_data, load_votes
import matplotlib.pyplot as plt
import seaborn as sns

def load_model_data(model_name, prompts, policies_to_ignore=None):
    """Load and process data for a specific model."""
    data = load_pairwise_data(model_name, prompts, policies_to_ignore=policies_to_ignore)
    data['model'] = model_name
    return data

def process_data(all_data, biographies_path="rep_biographies.jsonl"):
    """Process the combined data with biographies."""
    biographies = pd.read_json(biographies_path, lines=True)
    biographies['id_1'] = biographies['ID']
    
    joined = all_data.merge(biographies, on='id_1', how='left')
    simple = joined.copy()
    simple = simple.rename({"Political Affiliation": "political_affiliation"})
    simple["flipped"] = simple["flipped"].astype(int)
    simple = simple[simple.same_condition == True]
    
    return simple

def calculate_flip_stats(grouped_merged, models=None):
    """
    Calculate flip statistics for each policy and model.
    
    Args:
        grouped_merged: DataFrame containing the grouped and merged data
        models: List of model names to include. If None, uses all models.
    
    Returns:
        tuple: (stats DataFrame, policy order list)
    """
    # Calculate mean flip rate and confidence intervals
    stats = grouped_merged.groupby(['policy_id', 'statement', 'model'])[1].agg(['mean', 'std', 'count']).reset_index()
    stats['ci'] = 1.645 * (stats['std'] / np.sqrt(stats['count']))  # 95% CI = mean ± 1.96 * SE

    # Filter for specified models if provided
    if models is not None:
        stats = stats[stats['model'].isin(models)]

    # Calculate average flip rate across models for sorting
    avg_flip_rates = stats.groupby('policy_id')['mean'].mean().sort_values(ascending=True)
    policy_order = avg_flip_rates.index.tolist()
    
    return stats, policy_order

def create_flip_plot(stats, policy_order, save_path='../data/plots/flip_rates_comparison_plot.png'):
    """
    Create a horizontal bar chart showing flip rates for each policy.
    
    Args:
        stats: DataFrame containing the statistics
        policy_order: List of policy IDs in desired order
        save_path: Path to save the plot
    """
    # Create the plot
    plt.figure(figsize=(15, 10))
    sns.set_style("whitegrid")

    # Set up the positions for the bars
    n_policies = len(policy_order)
    x = np.arange(n_policies)
    width = 0.35 / len(stats['model'].unique())  # Adjust width based on number of models
    
    # Define colors for models
    colors = plt.cm.Set2(np.linspace(0, 1, len(stats['model'].unique())))
    color_mapping = dict(zip(stats['model'].unique(), colors))
    
    # Plot bars for each model
    for i, model in enumerate(stats['model'].unique()):
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def prepare_stats_table(stats):
    """
    Prepare statistics for table display.
    
    Args:
        stats: DataFrame containing the statistics
    
    Returns:
        tuple: (pivoted DataFrame, mean across models Series)
    """
    # Create mean and confidence interval column
    stats['mean_ci'] = stats.apply(lambda row: f"{row['mean']:.3f} ± {row['ci']:.3f}", axis=1)
    
    # Calculate mean across models for sorting
    mean_across_models = stats.groupby('statement')['mean'].mean().sort_values(ascending=False)
    
    # Pivot the data
    pivoted = stats.pivot(index=['statement'], columns='model', values='mean_ci').reset_index()
    pivoted = pivoted.set_index('statement').loc[mean_across_models.index].reset_index()
    
    return pivoted, mean_across_models

def create_latex_table(pivoted_df, models=None):
    """
    Create LaTeX table with bold formatting for top 3 policies with highest flip rates per model.
    
    Args:
        pivoted_df: DataFrame containing the pivoted data
        models: List of model names to include in the table. If None, uses all models in the data.
    """
    # Create a copy of the dataframe for formatting
    formatted_df = pivoted_df.copy()
    
    # Filter for specified models if provided
    if models is not None:
        formatted_df = formatted_df[['statement'] + models]
    
    # Format numbers to 2 decimal places and identify top 3 for each model
    for col in formatted_df.columns:
        if col != 'statement':
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
        columns=['statement'] + [col for col in formatted_df.columns if col != 'statement' and not col.endswith('_val')]
    )
    
    return latex_table

def analyze_demographics(simple_data):
    """Analyze demographic patterns in the data."""
    demographics = simple_data.columns[17:]  # Adjust index based on your data structure
    results = {}
    
    for demographic in demographics:
        results[demographic] = {
            'normalized': simple_data.groupby(demographic)["flipped"].value_counts(normalize=True).unstack(),
            'counts': simple_data.groupby(demographic)["flipped"].value_counts(normalize=False).unstack()
        }
    
    return results

#%%
# Configuration
prompts = ["prompt-0", "prompt-1", "prompt-2", "prompt-3", "prompt-4"]
policies_to_ignore = None

# Define models to use
models = {
    "GPT-4": "gpt-4o",
    "Claude": "claude-3-sonnet-v2",
    "Llama": "llama-3.2"
}

#%%
# Load data for all models
all_data_list = []
for display_name, model_name in models.items():
    data = load_model_data(model_name, prompts, policies_to_ignore)
    data['model'] = display_name  # Use display name instead of model name
    all_data_list.append(data)

#%%
# Combine and process the data
all_data = pd.concat(all_data_list, ignore_index=True)
simple = process_data(all_data)

#%%
# Create grouped data
grouped = simple.groupby(["policy_id", "model", "combined_prompt_name_1", "combined_prompt_name_2"])["flipped"].value_counts(normalize=True).unstack()
grouped = grouped.fillna(0).reset_index()

# Load policies and merge
policies = pd.read_json("../self_selected_policies.jsonl", lines=True)  
policies["policy_id"] = policies["id"]
grouped_merged = grouped.merge(policies, on="policy_id", how="left")

#%%
# Calculate statistics and create plot
stats, policy_order = calculate_flip_stats(grouped_merged)
# create_flip_plot(stats, policy_order)

#%%
# Prepare and create table
pivoted, mean_across_models = prepare_stats_table(stats)
latex_output = create_latex_table(pivoted)
print(latex_output)
# with open('../data/latex/flip_rates_table.tex', 'w') as f:
#     f.write(latex_output)

#%%
# Analyze demographics
demographic_results = analyze_demographics(simple)

#%%
# stats = grouped_merged.groupby(['policy_id', 'statement', 'model'])[1].agg(['mean', 'std', 'count']).reset_index()
# stats['ci'] = 1.645 * (stats['std'] / np.sqrt(stats['count']))  # 95% CI = mean ± 1.96 * SE

# # Filter for specified models if provided
# if models is not None:
#     stats = stats[stats['model'].isin(models)]

# # Calculate average flip rate across models for sorting
# avg_flip_rates = stats.groupby('policy_id')['mean'].mean().sort_values(ascending=True)
# policy_order = avg_flip_rates.index.tolist()
# # %%
# grouped[grouped.isna().any(axis=1)].groupby("policy_id").count()
# # %%
# single_data = []
# for model in models.keys():
#     single_model_data = load_votes(models[model], prompts, policies_to_ignore)\
#         .assign(model=model)
#     single_data.append(single_model_data)
# # %%
# single_data = pd.concat(single_data, ignore_index=True)
# # %%
# votes_per_policy = single_data.groupby(["policy_id", "model"])["vote"].value_counts(normalize=True).unstack()
# absolutes = votes_per_policy[(votes_per_policy["No"] == 1) | (votes_per_policy["Yes"] == 1)]
# # %%
# absolutes
# # %%
# votes_per_policy
# # %%
# grouped
# %%
