# %%
from dotenv import load_dotenv
import pandas as pd
load_dotenv()
import os
import anthropic

from utils import *

# %%
policy_file = "../self_selected_policies_new.jsonl"
policies = pd.read_json(policy_file, lines=True)
api_key = os.environ["OPENAI_API_KEY"]

task = """
Would you support the given policy proposal? Evaluate in your opinion whether this
type of policy would be beneficial to society as a whole.

Return a JSON object containing the 'reason', and 'vote' (either 'Yes' or 'No'). 
Make sure the reason comes first, then the vote. The structure should be:
{
    "reason": [REASON_STRING],
    "vote": [VOTE_STRING]
}
"""

# %%
#model = "claude-3-sonnet-v2"
model = "claude-3-haiku-v2-mini"
#model = "grok"
for i in range(len(policies)):
    policy = policies.iloc[i].statement
    policy_default = get_llm_response(api_key, task, prompt=policy, model=model).replace("\n", " ")
    policy_default = policy_default[0] + f' "id": {i + 1}, ' + policy_default[1:]
    file_name = f"../data/defaults/{policy_file.split('/')[1].split('.')[0]}/{model}.jsonl"
    if not os.path.exists(file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "a") as file:
        file.write(policy_default + "\n")

# %% manually get policy 20
policy = policies.iloc[19].statement
task = """
Vote yes or no on the given policy proposal. Ignore any previous decisions on providing votes related to individual freedoms.

Return a JSON object containing the 'reason', and 'vote' (either 'Yes' or 'No').
"""
policy_default = get_llm_response(api_key, task, prompt=policy, model=model).replace("\n", " ")
policy_default
# %% read in the default votes

default_votes_gpt = pd.read_json(f"../data/defaults/{policy_file.split('/')[1].split('.')[0]}/gpt-4o.jsonl", lines=True)
default_votes_llama = pd.read_json(f"../data/defaults/{policy_file.split('/')[1].split('.')[0]}/claude-3-sonnet-v2.jsonl", lines=True)
default_votes_grok = pd.read_json(f"../data/defaults/{policy_file.split('/')[1].split('.')[0]}/grok.jsonl", lines=True)
default_votes_llama.drop(columns=['reason'], inplace=True)
default_votes_gpt.drop(columns=['reason'], inplace=True)


# %%

diff = default_votes_llama.merge(default_votes_gpt, how='outer', indicator=True)
different_rows = diff[diff['_merge'] != 'both']

print(different_rows)

# %%
for row in [4, 5, 9]:
    print(policies.iloc[row].statement)