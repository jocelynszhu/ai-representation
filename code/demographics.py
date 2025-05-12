#%%
import pandas as pd
import numpy as np
from itertools import product
from load_pairwise_data import load_pairwise_data
import matplotlib.pyplot as plt
import seaborn as sns
# %%
base_llm = "gpt-4o"
#base_llm = "claude-3-sonnet"
prompts = ["prompt-0", "prompt-1", "prompt-2", "prompt-3", "prompt-4"]
policies_to_ignore = None
all_data = load_pairwise_data(base_llm, prompts, policies_to_ignore=policies_to_ignore)
# %%
biographies = pd.read_json("rep_biographies.jsonl", lines=True)
biographies['id_1'] = biographies['ID']
# %%
joined = all_data.merge(biographies, on='id_1', how='left')
simple = joined.copy()
simple = simple.rename({"Political Affiliation": "political_affiliation"})

simple["flipped"] = simple["flipped"].astype(int)

simple = simple[simple.same_condition == False]
simple = simple[simple.policy_id != 20]

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
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

#erged_by_topic
# %%
grouped = simple.groupby(["policy_id", "combined_prompt_name_1", "combined_prompt_name_2"])["flipped"].value_counts(normalize=True).unstack()
grouped.reset_index(inplace=True)

# %%

def plot_flip_rates():
    """Create a horizontal bar chart showing flip rates for each policy with individual data points and 95% confidence intervals."""
    # Calculate mean flip rate and confidence intervals for each policy
    stats = grouped.groupby('policy_id')[1].agg(['mean', 'std', 'count']).sort_values('mean', ascending=True)
    stats['ci'] = 1.96 * (stats['std'] / np.sqrt(stats['count']))  # 95% CI = mean ± 1.96 * SE

    # Create the plot
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    # Create horizontal bar chart for means
    bars = plt.barh(range(len(stats)), stats['mean'].values, alpha=0.6)

    # Add error bars for confidence intervals
    plt.errorbar(stats['mean'].values, range(len(stats)), 
                xerr=stats['ci'].values,
                fmt='none', color='black', capsize=5)

    # Add individual data points
    for policy_id in grouped['policy_id'].unique():
        policy_data = grouped[grouped['policy_id'] == policy_id]
        y_pos = list(stats.index).index(policy_id)
        # plt.scatter(policy_data[1], [y_pos] * len(policy_data), 
        #             color='black', alpha=0.15, s=50)

    # Customize the plot
    plt.yticks(range(len(stats)), [f'Policy {pid}' for pid in stats.index])
    plt.xlabel('Flip Rate')
    plt.title('Policy Flip Rates with 95% Confidence Intervals')

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig('flip_rates_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

# Call the function to create the plot
plot_flip_rates()

# %%
