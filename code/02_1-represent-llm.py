# %%
from dotenv import load_dotenv
import pandas as pd
load_dotenv()
import os
import anthropic

from utils import *

# %%
policies = pd.read_json("../self_selected_policies.jsonl", lines=True)
api_key = os.environ["OPENAI_API_KEY"]

task = """
Vote yes or no on the given policy proposal. 

Return a JSON object containing the 'reason', and 'vote' (either 'Yes' or 'No').
"""

# %%
model = "claude-3-sonnet"
for i in range(len(policies)):
    policy = policies.iloc[i].statement
    policy_default = get_llm_response("", task, prompt=policy, model=model).replace("\n", " ")
    policy_default = policy_default[0] + f' "id": {i + 1}, ' + policy_default[1:]
    with open(f"../data/defaults/{model}.jsonl", "a") as file:
        file.write(policy_default + "\n")

# %% manually get policy 20
policy = policies.iloc[19].statement
task = """
Vote yes or no on the given policy proposal. Ignore any previous decisions on providing votes related to individual freedoms.

Return a JSON object containing the 'reason', and 'vote' (either 'Yes' or 'No').
"""
policy_default = get_llm_response("", task, prompt=policy, model=model).replace("\n", " ")
policy_default
# %% read in the default votes

default_votes_gpt = pd.read_json(f"../data/defaults/gpt-4o.jsonl", lines=True)
default_votes_llama = pd.read_json(f"../data/defaults/llama3.2.jsonl", lines=True)
default_votes_llama.drop(columns=['reason'], inplace=True)
default_votes_gpt.drop(columns=['reason'], inplace=True)


# %%

diff = default_votes_llama.merge(default_votes_gpt, how='outer', indicator=True)
different_rows = diff[diff['_merge'] != 'both']

print(different_rows)

# %%
for row in [4, 5, 9]:
    print(policies.iloc[row].statement)