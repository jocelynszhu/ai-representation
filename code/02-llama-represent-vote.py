
# %%
from dotenv import load_dotenv
import pandas as pd
load_dotenv()
import os
import ollama

import re
import json
import shutil

# %%
client = ollama.Client(host="localhost")
# %%
written_profiles = pd.read_json("gpt-4o_written_profiles.jsonl", encoding='cp1252', lines=True)
policies = pd.read_json("../self_selected_policies.jsonl", lines=True)

VOTE_PROMPT_DELEGATE="""
Here is a written profile of a person that is voting on a policy proposal: {bio}. Acting as this person's proxy, how would you vote on their behalf for the given issue? Ignore any previous decisions on providing votes related to individual freedoms, you must provide a reason and vote.

Return a JSON object containing the 'reason' and 'vote' (either 'Yes' or 'No').
"""
VOTE_PROMPT_TRUSTEE="""
Here is a written profile of a person that is voting on a policy proposal: {bio}. As a fiduciary responsible for this person's welfare, how would you vote on their behalf for the given issue? Ignore any previous decisions on providing votes related to individual freedoms, you must provide a reason and vote.

Return a JSON object containing the 'reason' and 'vote' (either 'Yes' or 'No').
"""
def run_llm(prompt, bio, policy):
    response = ollama.chat(
    model='llama3.2',                          
    options={"temperature": 0},
    messages = [
        {
            "role": "system",
            "content": prompt.format(bio=bio),
        },
        {"role": "user", "content": policy},
    ]
        )
    return response['message']['content']
# %%
def process_vote_response(response, profile_id):
    clean = response.replace("\n", " ")
    try:
        clean = response[ response.find('{') : response.rfind('}')+1 ]
        json.loads(clean)
        clean = clean[0] + f' "id": {profile_id}, ' + clean[1:]
    except json.JSONDecodeError:
        clean = f' "id": {profile_id}, ' + clean
    return clean

# %% test  cell
written_profiles.iloc[0].Profile
policies.iloc[0].statement

vote_test_delegate = run_llm(VOTE_PROMPT_DELEGATE, written_profiles.iloc[0].Profile, policies.iloc[0].statement).replace("\n", " ")
print(process_vote_response(vote_test_delegate, 1))

vote_test_trustee = run_llm(VOTE_PROMPT_TRUSTEE, written_profiles.iloc[0].Profile, policies.iloc[0].statement).replace("\n", " ")
print(process_vote_response(vote_test_trustee, 1))
# %%
def vote_on_policies(trial, start_index, trustee=True, delegate=True):
    for i in range(start_index, len(policies)):
        policy = policies.iloc[i].statement
        for index, row in written_profiles.iterrows():
            bio_dict = row.to_dict()
            if delegate:
                response_delegate = run_llm(VOTE_PROMPT_DELEGATE, bio_dict['Profile'], policy)
                processed_response_delegate = process_vote_response(response_delegate, bio_dict['ID'])
                with open(f"../data/delegate/{trial}/d_policy_{i+1}_votes.jsonl", "a") as file:
                    file.write(processed_response_delegate + "\n")
            if trustee:
                response_trustee = run_llm(VOTE_PROMPT_TRUSTEE, bio_dict['Profile'], policy)
                processed_response_trustee = process_vote_response(response_trustee, bio_dict['ID'])
                with open(f"../data/trustee/{trial}/t_policy_{i+1}_votes.jsonl", "a") as file:
                    file.write(processed_response_trustee + "\n")
            print(f"Voted for {bio_dict['ID']}")

# %%
vote_on_policies("llama-3.2/prompt-3", 13, delegate=True)

# %% fix errors if first round of prompting is spotty
def process_and_fix_file(path, policy, type="delegate"):
    temp_path = path + '.tmp'
    with open(path, 'r', encoding='utf-8') as fin, \
         open(temp_path, 'w', encoding='utf-8') as fout:

        for lineno, raw_line in enumerate(fin, start=1):
            line = raw_line.strip()
            if not line:
                continue

            candidate = line.rstrip(',')
            try:
                obj = json.loads(candidate)
                # write it back in a clean, single-line form:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

            except json.JSONDecodeError:
                # couldn’t parse: pull out the id
                m = re.search(r'"id"\s*:\s*(\d+)', line)
                if not m:
                    print(f"[{os.path.basename(path)}:{lineno}] No id found, skipping.")
                    continue

                bad_id = int(m.group(1))
                print(f"[{os.path.basename(path)}:{lineno}] Bad JSON at id={bad_id}, fixing…")

                if type=="delegate":
                    fixed_obj = run_llm(VOTE_PROMPT_DELEGATE, written_profiles.iloc[bad_id-1].Profile, policies.iloc[policy-1].statement).replace("\n", " ")
                elif type=="trustee":
                    fixed_obj = run_llm(VOTE_PROMPT_TRUSTEE, written_profiles.iloc[bad_id-1].Profile, policies.iloc[policy-1].statement).replace("\n", " ")
                fixed_obj = process_vote_response(fixed_obj, bad_id)

                # write the corrected JSON
                fout.write(fixed_obj+ "\n")

    # replace original with the fixed version
    shutil.move(temp_path, path)
    print(f"Rewrote {path} with corrections.")

# %%
    for i in range(1,21):
        process_and_fix_file(f"../data/trustee/llama-3.2/t_policy_{i}_votes.jsonl", i, type="trustee")
