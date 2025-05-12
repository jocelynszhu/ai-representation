#%%
import numpy as np
#from scipy.stats import mcnemar
from statsmodels.stats.contingency_tables import mcnemar
import pandas as pd
#%%
def prepare_mcnemar_data(data):
    """
    Prepare data for McNemar's test.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with columns: participant_id, policy_id, condition, vote
        
    Returns:
    --------
    tuple
        (b, c) where:
        b = number of cases where vote was Yes in condition 1 but No in condition 2
        c = number of cases where vote was No in condition 1 but Yes in condition 2
    """
    # Pivot the data to get votes for both conditions side by side
    pivoted = data.pivot_table(
        index=['participant_id', 'policy_id'],
        columns='condition',
        values='vote',
        aggfunc='first'
    ).reset_index()
    
    # Calculate b and c
    b = ((pivoted[1] == 1) & (pivoted[2] == 0)).sum()
    c = ((pivoted[1] == 0) & (pivoted[2] == 1)).sum()
    
    return b, c

def run_mcnemars_test(data):
    """
    Run McNemar's test on the voting data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with columns: participant_id, policy_id, condition, vote
        
    Returns:
    --------
    tuple
        (statistic, pvalue) from McNemar's test
    """
    b, c = prepare_mcnemar_data(data)
    
    # Run McNemar's test
    statistic, pvalue = mcnemar([[0, b], [c, 0]], exact=True)
    
    return statistic, pvalue

def analyze_voting_patterns(data):
    """
    Analyze voting patterns between conditions.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with columns: participant_id, policy_id, condition, vote
        
    Returns:
    --------
    dict
        Dictionary containing various statistics about the voting patterns
    """
    b, c = prepare_mcnemar_data(data)
    
    # Calculate total number of changes
    total_changes = b + c
    
    # Calculate percentage of changes
    total_votes = len(data) // 2  # Since each vote appears twice (once per condition)
    percent_changes = (total_changes / total_votes) * 100
    
    # Calculate direction of changes
    percent_to_yes = (c / total_changes) * 100 if total_changes > 0 else 0
    percent_to_no = (b / total_changes) * 100 if total_changes > 0 else 0
    
    return {
        'total_changes': total_changes,
        'percent_changes': percent_changes,
        'percent_to_yes': percent_to_yes,
        'percent_to_no': percent_to_no
    }

#%%
all_data = []
model = 'claude-3-sonnet'
prompts = ['prompt-1', 'prompt-2', 'prompt-3', 'prompt-4']
for prompt in prompts:
    for policy in range(1, 21):
        # Load data
        print(f'Processing policy {policy} of 20')
        filename_trustee = f'../data/trustee/{model}/{prompt}/t_policy_{policy}_votes.jsonl'
        filename_delegate = f'../data/delegate/{model}/{prompt}/d_policy_{policy}_votes.jsonl'
        print(filename_trustee  )
        print(filename_delegate)
        data_trustee = pd.read_json(filename_trustee, encoding='cp1252', lines=True).assign(condition = 'trustee', policy_id = policy, prompt = prompt)
        data_delegate = pd.read_json(filename_delegate, encoding='cp1252', lines=True).assign(condition = 'delegate', policy_id = policy, prompt = prompt)
        data = pd.concat([data_trustee, data_delegate])
        print(pd.DataFrame(data))
        print(pd.DataFrame(data_delegate))
        all_data.append(data)
# %%
print(len(all_data))
# %%
all_data_df = pd.concat(all_data)
all_data_df = all_data_df.replace({'Yes': 1, 'No': 0})
# %%
# %%
all_data_df
# %%
ids = all_data_df.id.unique()
policies = all_data_df.policy_id.unique()
# %%
# for id in ids:
#     for policy in policies:
#         data = all_data_df[all_data_df['id'] == id]
#         data = data[data['policy_id'] == policy]
#         print(data)
#         break
# # %%
# data
# %%
# Create a function to compare votes across prompt pairs
def compare_prompt_pairs(data, condition=None):
    """Compare votes between all pairs of prompts for a given condition"""
    prompts = ['prompt-1', 'prompt-2', 'prompt-3', 'prompt-4']
    results = []
    
    # Filter by condition if specified
    if condition:
        data = data[data['condition'] == condition]
    
    # Generate all prompt pairs
    for i, prompt1 in enumerate(prompts):
        for prompt2 in prompts[i+1:]:
            # For each participant and policy
            for id in data.id.unique():
                for policy in data.policy_id.unique():
                    # Get votes for this participant-policy combo in both prompts
                    vote1 = data[(data['id'] == id) & 
                               (data['policy_id'] == policy) &
                               (data['prompt'] == prompt1)]['vote'].values
                    
                    vote2 = data[(data['id'] == id) & 
                               (data['policy_id'] == policy) &
                               (data['prompt'] == prompt2)]['vote'].values
                    
                    # Only compare if we have both votes
                    if len(vote1) > 0 and len(vote2) > 0:
                        flipped = int(vote1[0] != vote2[0])
                        results.append({
                            'id': id,
                            'policy_id': policy,
                            'prompt1': prompt1,
                            'prompt2': prompt2,
                            'vote1': vote1[0],
                            'vote2': vote2[0],
                            'flipped': flipped,
                            'condition': condition if condition else 'between'
                        })
    
    return pd.DataFrame(results)
#%%
# Within-condition comparisons
trustee_comparisons = compare_prompt_pairs(all_data_df, 'trustee')
delegate_comparisons = compare_prompt_pairs(all_data_df, 'delegate')
#%%
# Between-condition comparisons
between_comparisons = []
prompts = ['prompt-1', 'prompt-2', 'prompt-3', 'prompt-4']
for i, prompt1 in enumerate(prompts):
    for prompt2 in prompts[i+1:]:
        for id in all_data_df.id.unique():
            for policy in all_data_df.policy_id.unique():
            # Get trustee and delegate votes
            trustee_vote = all_data_df[(all_data_df['id'] == id) &
                                     (all_data_df['policy_id'] == policy) &
                                     (all_data_df['prompt'] == prompt) &
                                     (all_data_df['condition'] == 'trustee')]['vote'].values
            
            delegate_vote = all_data_df[(all_data_df['id'] == id) &
                                      (all_data_df['policy_id'] == policy) &
                                      (all_data_df['prompt'] == prompt) &
                                      (all_data_df['condition'] == 'delegate')]['vote'].values
            
            if len(trustee_vote) > 0 and len(delegate_vote) > 0:
                flipped = int(trustee_vote[0] != delegate_vote[0])
                between_comparisons.append({
                    'id': id,
                    'policy_id': policy,
                    'prompt': prompt,
                    'trustee_vote': trustee_vote[0],
                    'delegate_vote': delegate_vote[0],
                    'flipped': flipped
                })

between_comparisons_df = pd.DataFrame(between_comparisons)
#%%
# Calculate summary statistics
within_trustee_flip_rate = trustee_comparisons['flipped'].mean()
within_delegate_flip_rate = delegate_comparisons['flipped'].mean()
between_flip_rate = between_comparisons_df['flipped'].mean()

print(f"Within-trustee flip rate: {within_trustee_flip_rate:.3f}")
print(f"Within-delegate flip rate: {within_delegate_flip_rate:.3f}")
print(f"Between-condition flip rate: {between_flip_rate:.3f}")

# %%
import statsmodels.formula.api as smf

model = smf.logit("flipped ~ C(condition)", data=between_comparisons_df).fit()
print(model.summary())
# %%
