# %%
import pandas as pd
import numpy as np
# %%
policies = pd.read_json("../self_selected_policies.jsonl", lines=True)

# %%
vote_stats_list = []
trial = "gpt-4o/prompt-3"
for i in range(1, 21):
    print(i)
    
    #votes_trustee = pd.read_json(f"../data/delegate/gpt-4o/prompt-3/d_policy_{i}_votes.jsonl",encoding='cp1252', lines=True)
    #votes_delegate = pd.read_json(f"../data/delegate/gpt-4o/prompt-3/d_policy_{i}_votes.jsonl",encoding='cp1252', lines=True)
    votes_trustee = pd.read_json(f"../data/trustee/{trial}/t_policy_{i}_votes.jsonl",encoding='cp1252', lines=True)
    votes_delegate = pd.read_json(f"../data/delegate/{trial}/d_policy_{i}_votes.jsonl",encoding='cp1252', lines=True)

    # Count the occurrences of each vote type
    trustee_counts = votes_trustee['vote'].value_counts()
    delegate_counts = votes_delegate['vote'].value_counts()
    
    # Extract counts for 'Yes' and 'No' votes, defaulting to 0 if not present
    t_yes_count = trustee_counts.get('Yes', 0)
    t_no_count = trustee_counts.get('No', 0)
    d_yes_count = delegate_counts.get('Yes', 0)
    d_no_count = delegate_counts.get('No', 0)
    
    # Append the counts along with the filename to the list
    vote_stats_list.append({
        'policy_id': i,
        'policy': policies.iloc[i-1].statement,
        'trustee_yes': t_yes_count,
        'trustee_no': t_no_count,
        'delegate_yes': d_yes_count,
        'delegate_no': d_no_count
    })
    
vote_stats = pd.DataFrame(vote_stats_list)
vote_stats

# %% votes that match default vote

# model_default = pd.read_json(f"../data/defaults/{trial}.jsonl", lines=True)
# model_default = model_default.rename(columns={'id': 'policy_id', 'vote': 'vote_ref'})

# default_check= model_default.merge(vote_stats, on='policy_id')

# default_check['%_delegate_match'] = np.where(
#     default_check['vote_ref'].str.lower() == 'yes',
#     default_check['delegate_yes'] / 100,
#     default_check['delegate_no']  / 100
# )
# default_check['%_trustee_match'] = np.where(
#     default_check['vote_ref'].str.lower() == 'yes',
#     default_check['trustee_yes'] / 100,
#     default_check['trustee_no']  / 100
# )
# default_check['%_delegate_match'].mean(), default_check['%_trustee_match'].mean()
# # %%
# vote_stats.to_csv("../data/vote_stats.csv", index=False)

# %% find flipped cases

flip_votes_list = []

for i in range(1, 21):
    votes_trustee = pd.read_json(f"../data/delegate/gpt-4o/prompt-3/d_policy_{i}_votes.jsonl",encoding='cp1252', lines=True)
    votes_delegate = pd.read_json(f"../data/delegate/gpt-4o/prompt-4/d_policy_{i}_votes.jsonl",encoding='cp1252', lines=True)
    #votes_trustee = pd.read_json(f"../data/trustee/{trial}/t_policy_{i}_votes.jsonl",encoding='cp1252', lines=True)
    #votes_delegate = pd.read_json(f"../data/delegate/{trial}/d_policy_{i}_votes.jsonl",encoding='cp1252', lines=True)
    votes_trustee['source'] = 'trustee'
    votes_delegate['source'] = 'delegate'
    votes_trustee['idx'] = range(len(votes_trustee))
    votes_delegate['idx'] = range(len(votes_delegate))

    # Merge on a shared column — adjust if needed
    
    merged = pd.merge(votes_trustee, votes_delegate, how='inner', on='idx', suffixes=('_trustee', '_delegate'))
    flipped = merged[
        ((merged['vote_trustee'] == 'Yes') & (merged['vote_delegate'] == 'No')) |
        ((merged['vote_trustee'] == 'No') & (merged['vote_delegate'] == 'Yes'))
    ]
    # Add to list
    for _, row in flipped.iterrows():
        flip_votes_list.append({
            'participant_idx': row['idx'],
            'policy_id': i,
            'policy': policies.iloc[i-1].statement,
            'vote_trustee': row['vote_trustee'],
            'reason_trustee': row['reason_trustee'],
            'vote_delegate': row['vote_delegate'],
            'reason_delegate': row['reason_delegate']
        })

# Create final DataFrame of flipped votes
flipped = pd.DataFrame(flip_votes_list)
print(len(flipped)/(20*len(merged)))
flipped
#%%
#votes_trustee
# %% check flipped counts if default to model

merged = flipped.merge(model_default, on='policy_id')

merged['vote_match'] = merged['vote_trustee'] == merged['vote_ref']
match_count = merged['vote_match'].sum()

print("Number of matching votes trustee:", match_count)

# %%
biographies = pd.read_json("rep_biographies.jsonl", lines=True)
biographies['participant_idx'] = range(len(biographies))
flipped_biography = flipped.merge(biographies, on='participant_idx', how='left')
flipped_biography
# %%
flipped.to_csv("../data/flipped_votes.csv", index=False)
flipped_biography.to_csv("../data/flipped_votes_biography.csv", index=False)

# %%
flipped_biography["Political Affiliation"].value_counts()
# %%
biographies["Political Affiliation"].value_counts()
# %%
flipped_biography["Political Affiliation"].value_counts()
# %%

