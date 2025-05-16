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
    # Keep both conditions
    return simple

def calculate_flip_stats(grouped_merged, models=None):
    """
    Calculate flip statistics for each policy, model, and condition.
    
    Args:
        grouped_merged: DataFrame containing the grouped and merged data
        models: List of model names to include. If None, uses all models.
    
    Returns:
        tuple: (stats DataFrame, policy order list)
    """
    # Calculate mean flip rate and confidence intervals
    stats = grouped_merged.groupby(['policy_id', 'statement', 'model', 'same_condition'])[1].agg(['mean', 'std', 'count']).reset_index()
    stats['ci'] = 1.645 * (stats['std'] / np.sqrt(stats['count']))  # 95% CI = mean ± 1.96 * SE

    # Filter for specified models if provided
    if models is not None:
        stats = stats[stats['model'].isin(models)]

    # Calculate average flip rate across models for sorting
    avg_flip_rates = stats.groupby('policy_id')['mean'].mean().sort_values(ascending=True)
    policy_order = avg_flip_rates.index.tolist()
    
    return stats, policy_order

def prepare_stats_table(stats):
    """
    Prepare statistics for table display showing only the differences between conditions.
    Sort by average of Claude and GPT-4 values.
    
    Args:
        stats: DataFrame containing the statistics
    
    Returns:
        tuple: (pivoted DataFrame, mean across models Series)
    """
    # Create hierarchical structure
    # First pivot by model and same_condition
    pivoted = stats.pivot_table(
        index=['statement'],
        columns=['model', 'same_condition'],
        values='mean'
    ).reset_index()
    
    # Calculate the difference between False and True conditions for each model
    diff_df = pd.DataFrame()
    diff_df['statement'] = pivoted['statement']
    
    for model in stats['model'].unique():
        false_col = (model, False)
        true_col = (model, True)
        if false_col in pivoted.columns and true_col in pivoted.columns:
            # Calculate difference
            diff = (pivoted[false_col] - pivoted[true_col]).round(3)
            diff_df[model] = diff
    
    # Calculate average of Claude and GPT-4 for sorting
    avg_claude_gpt = (diff_df['Claude'] + diff_df['GPT-4']).div(2)
    
    # Sort by average and keep statement as a column
    diff_df = diff_df.sort_values(by=['Claude', 'GPT-4'], ascending=False)
    
    return diff_df, avg_claude_gpt

def create_latex_table(pivoted_df, models=None):
    """
    Create LaTeX table showing differences between conditions for each model.
    Bold the top three largest values in each model column.
    
    Args:
        pivoted_df: DataFrame containing the differences
        models: List of model names to include in the table. If None, uses all models in the data.
    """
    # Create a copy of the dataframe for formatting
    formatted_df = pivoted_df.copy()
    
    # Format numbers and bold top 3 for each model
    for col in formatted_df.columns:
        if col != 'statement':
            # First get top 3 policies using original values
            top_3_policies = formatted_df.nlargest(3, col)['statement'].tolist()
            
            # Format all values to exactly 3 decimal places
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.3f}")
            
            # Add bold formatting for top 3 policies
            for policy in top_3_policies:
                mask = formatted_df['statement'] == policy
                formatted_df.loc[mask, col] = '\\textbf{' + formatted_df.loc[mask, col] + '}'
    
    # Generate LaTeX table
    latex_table = formatted_df.to_latex(
        index=False,
        escape=False,
        caption='Differences in flip rates between conditions (False - True)',
        label='tab:flip_rates_diff',
        columns=['statement'] + [col for col in formatted_df.columns if col != 'statement']
    )
    
    # Add footnotesize command and make statement column a paragraph
    latex_table = latex_table.replace('\\begin{table}', '\\begin{table}\n\\footnotesize')
    latex_table = latex_table.replace('\\begin{tabular}', '\\begin{tabular}{p{4cm}' + 'c' * (len(formatted_df.columns) - 1) + '}')
    
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
# # Load data for all models
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
grouped = simple.groupby(["policy_id", "model", "combined_prompt_name_1", "combined_prompt_name_2", "same_condition"])["flipped"].value_counts(normalize=True).unstack()
grouped = grouped.fillna(0).reset_index()

# Load policies and merge
policies = pd.read_json("../self_selected_policies.jsonl", lines=True)  
policies["policy_id"] = policies["id"]
grouped_merged = grouped.merge(policies, on="policy_id", how="left")

# #%%
# # Calculate statistics and create plot
stats, policy_order = calculate_flip_stats(grouped_merged)
#create_flip_plot(stats, policy_order)

#%%
# Prepare and create table
pivoted, mean_across_models = prepare_stats_table(stats)
#%%
latex_output = create_latex_table(pivoted)
#%%1
print(latex_output)
with open('../data/latex/flip_rates_table_excess.tex', 'w') as f:
    f.write(latex_output)

#%%
# Analyze demographics
demographic_results = analyze_demographics(simple)


# %%
