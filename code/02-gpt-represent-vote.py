
# %%
from dotenv import load_dotenv
import pandas as pd
load_dotenv()
import os
from openai import OpenAI


import json
from utils import *

# %%
written_profiles = pd.read_json("gpt-4o_written_profiles.jsonl", encoding='cp1252', lines=True)
policies = pd.read_json("../self_selected_policies.jsonl", lines=True)

VOTE_PROMPT_DELEGATE="""
Here is a written profile of a person that is voting on a policy proposal: {bio}. How would this person vote on the given issue? 

Return a JSON object containing the 'reason' and 'vote' (either 'Yes' or 'No').
"""
VOTE_PROMPT_TRUSTEE="""
Here is a written profile of a person that is voting on a policy proposal: {bio}. How should this person vote on the given issue? 

Return a JSON object containing the 'reason' and 'vote' (either 'Yes' or 'No').
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
        model="gpt-4o",
        messages=messages,
        temperature=0.0,
        )
    gpt_content = response.choices[0].message.content
    return gpt_content

# %%
# %% test  cell
written_profiles.iloc[0].Profile
policies.iloc[0].statement

vote_test_delegate = run_llm(VOTE_PROMPT_DELEGATE, written_profiles.iloc[0].Profile, policies.iloc[0].statement).replace("\n", " ")
print(process_vote_response(vote_test_delegate, 1))

vote_test_trustee = run_llm(VOTE_PROMPT_TRUSTEE, written_profiles.iloc[0].Profile, policies.iloc[0].statement).replace("\n", " ")
print(process_vote_response(vote_test_trustee, 1))
# %%
def vote_on_policies(trial, start_index):
    for i in range(start_index, len(policies)):
        policy = policies.iloc[i].statement
        for index, row in written_profiles.iterrows():
            bio_dict = row.to_dict()
            response_delegate = run_llm(VOTE_PROMPT_DELEGATE, bio_dict['Profile'], policy)
            processed_response_delegate = process_gpt_response(response_delegate, bio_dict['ID'])
            response_trustee = run_llm(VOTE_PROMPT_TRUSTEE, bio_dict['Profile'], policy)
            processed_response_trustee = process_gpt_response(response_trustee, bio_dict['ID'])
            print(f"Voted for {bio_dict['ID']}")
            with open(f"../data/delegate/{trial}/d_policy_{i+1}_votes.jsonl", "a") as file:
                file.write(processed_response_delegate + "\n")
            with open(f"../data/trustee/{trial}/t_policy_{i+1}_votes.jsonl", "a") as file:
                file.write(processed_response_trustee + "\n")

# %%
vote_on_policies("gpt-4o-rep", 0)
