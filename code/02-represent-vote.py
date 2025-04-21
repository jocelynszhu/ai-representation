
# %%
from dotenv import load_dotenv
import pandas as pd
load_dotenv()
import os
from openai import OpenAI


import re
import json
import random

# %%
written_profiles = pd.read_json("written_profiles.jsonl", encoding='cp1252', lines=True)
policies = pd.read_json("../self_selected_policies.jsonl", lines=True)

VOTE_PROMPT_DELEGATE="""
Here is a written profile of a person that is voting on a policy proposal: {bio}. How would this person vote on the given issue? 

Respond with a JSON object containing 'vote' (either 'Yes' or 'No') and 'reason'.
"""
VOTE_PROMPT_TRUSTEE="""
Here is a written profile of a person that is voting on a policy proposal: {bio}. How would you vote on the given issue for this person's best interest? 

Respond with a JSON object containing 'vote' (either 'Yes' or 'No') and 'reason'.
"""

def run_llm(prompt, profile, policy):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    messages = [
        {
            "role": "system",
            "content": prompt.format(bio=profile),
        },
        {"role": "user", "content": policy},
    ]
    

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.0,
        )
    gpt_content = response.choices[0].message.content
    
    return gpt_content.strip()

# %% test  cell
written_profiles.iloc[0].Profile
policies.iloc[0].statement

vote_test_delegate = run_llm(VOTE_PROMPT_DELEGATE, written_profiles.iloc[0].Profile, policies.iloc[0].statement)
print(vote_test_delegate)

vote_test_trustee = run_llm(VOTE_PROMPT_TRUSTEE, written_profiles.iloc[0].Profile, policies.iloc[0].statement)
print(vote_test_trustee)
# %%
for i in range(15, len(policies)):
    policy = policies.iloc[i].statement
    for index, row in written_profiles.iterrows():
        bio_dict = row.to_dict()
        response_delegate = run_llm(VOTE_PROMPT_DELEGATE, bio_dict['Profile'], policy)
        response_trustee = run_llm(VOTE_PROMPT_TRUSTEE, bio_dict['Profile'], policy)
        print(f"Voted for {bio_dict['UUID']}")
        with open(f"../data/delegate/d_policy_{i+1}_votes.jsonl", "a") as file:
            file.write(response_delegate.strip() + "\n")
        with open(f"../data/trustee/t_policy_{i+1}_votes.jsonl", "a") as file:
            file.write(response_trustee.strip() + "\n")

# %% fix formatting
def fix_malformed_jsonl(input_path, output_path):
    buffer = ""
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line in infile:
            buffer += line.strip()  # remove extra whitespace/newlines
            try:
                obj = json.loads(buffer)
                outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")
                buffer = ""  # reset buffer after success
            except json.JSONDecodeError:
                buffer += " "  # allow whitespace to separate continued lines

        # Handle any leftover buffer at the end
        if buffer.strip():
            try:
                obj = json.loads(buffer)
                outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")
            except json.JSONDecodeError:
                print("Final buffer could not be parsed. Skipping.")


# %%
for i in range(1, 21):
    input_path = f"../data/delegate/d_policy_{i}_votes.jsonl"
    output_path = f"../data/delegate/d_policy_{i}_votes_formatted.jsonl"
    fix_malformed_jsonl(input_path, output_path)

    input_path = f"../data/trustee/t_policy_{i}_votes.jsonl"
    output_path = f"../data/trustee/t_policy_{i}_votes_formatted.jsonl"
    fix_malformed_jsonl(input_path, output_path)


# %% get statistics for all policies to dataframe
vote_stats_list = []

for i in range(1, 21):
    votes_trustee = pd.read_json(f"../data/trustee/t_policy_{i}_votes_formatted.jsonl",encoding='cp1252', lines=True)
    votes_delegate = pd.read_json(f"../data/delegate/d_policy_{i}_votes_formatted.jsonl",encoding='cp1252', lines=True)

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
        'policy': policies.iloc[i-1].statement,
        'trustee_yes': t_yes_count,
        'trustee_no': t_no_count,
        'delegate_yes': d_yes_count,
        'delegate_no': d_no_count
    })
    
vote_stats = pd.DataFrame(vote_stats_list)

# %%
vote_stats.to_csv("../data/vote_stats.csv", index=False)

# %% find flipped cases

flip_votes_list = []

for i in range(1, 21):
    votes_trustee = pd.read_json(f"../data/trustee/t_policy_{i}_votes_formatted.jsonl", encoding='cp1252', lines=True)
    votes_delegate = pd.read_json(f"../data/delegate/d_policy_{i}_votes_formatted.jsonl", encoding='cp1252', lines=True)
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

# %%
biographies = pd.read_json("biographies.jsonl", lines=True)
biographies['participant_idx'] = range(len(biographies))
flipped_biography = flipped.merge(biographies, on='participant_idx', how='left')

# %%
flipped.to_csv("../data/flipped_votes.csv", index=False)
flipped_biography.to_csv("../data/flipped_votes_biography.csv", index=False)

# %%
biographies["Political Affiliation"].value_counts()

# %%







# %% 
votes_trustee = pd.read_json("../data/trustee/policy_1_votes.jsonl", lines=True)
votes_delegate = pd.read_json("../data/delegate/policy_1_votes.jsonl", lines=True)

num_votes = len(votes_trustee)
delegate_yes = 0
trustee_yes = 0
num_mismatches = 0
for i in range(num_votes):
    if votes_trustee.iloc[i].vote == "yes":
        trustee_yes += 1
    if votes_delegate.iloc[i].vote == "yes":
        delegate_yes += 1
    if votes_trustee.iloc[i].vote != votes_delegate.iloc[i].vote:
        num_mismatches += 1

print(policies.iloc[0].statement)
print(f"Trustee yes votes: {trustee_yes}") 
print(f"Delegate yes votes: {delegate_yes}")
print(f"Number of mismatches: {num_mismatches}")