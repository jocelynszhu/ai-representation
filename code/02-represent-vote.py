
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

Please return in JSONL format with two key-value pairs, one is the person’s reasoning on how they decided what to vote, and second is their vote, yes or no.

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
        response_format={ "type": "json_object" }
    )
    gpt_content = response.choices[0].message.content
    
    return gpt_content

# %%
written_profiles.iloc[0].Profile
policies.iloc[0].statement

vote_test = run_llm(VOTE_PROMPT_DELEGATE, written_profiles.iloc[0].Profile, policies.iloc[0].statement)
vote_test
# %%
policy = policies.iloc[0].statement
for index, row in written_profiles.iterrows():
    bio_dict = row.to_dict()
    response = run_llm(VOTE_PROMPT_DELEGATE, bio_dict['Profile'], policy)
    print(f"Voted for {bio_dict['UUID']}")

    with open("../data/delegate/policy_1_votes.jsonl", "a") as file:
        file.write(response + "\n")