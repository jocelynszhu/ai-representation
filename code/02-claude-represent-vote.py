#!/usr/bin/env python3

import argparse
from dotenv import load_dotenv
import pandas as pd
import os
import json
import anthropic
from utils import *
load_dotenv()
def parse_args():
    parser = argparse.ArgumentParser(description='Run voting simulation with Claude model')
    parser.add_argument('--prompt-num', type=int, required=True, help='Number of the prompt to use from prompts.json')
    parser.add_argument('--trial', type=str, required=True, help='Name of the trial (e.g., claude-3-sonnet/prompt-1)')
    parser.add_argument('--start-index', type=int, default=0, help='Starting index for policies (default: 0)')
    return parser.parse_args()

def load_prompts():
    with open('../prompts.json', 'r') as f:
        return json.load(f)

def run_claude(prompt, profile, policy):
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    
    # messages = [
    #     {
    #         "role": "user",
    #         "content": f"{prompt.format(bio=profile)}\n\nPolicy proposal: {policy}"
    #     }
    # ]
    messages = [
        {"role": "user", "content": policy},
    ]
    system = prompt.format(bio=profile)
   # print(system)
   # print(messages)
    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=1024,
        messages=messages,
        system = system,
        temperature=0.0
    )
   # print(response)
    #raise Exception("Stop here")
    return response.content[0].text

def process_vote_response(response, profile_id):
    clean = response.replace("\n", " ")
    try:
        clean = response[response.find('{'):response.rfind('}')+1]
        json.loads(clean)
        clean = clean[0] + f' "id": {profile_id}, ' + clean[1:]
    except json.JSONDecodeError:
        clean = f' "id": {profile_id}, ' + clean
    return clean

def vote_on_policies(trial, start_index, delegate_prompt, trustee_prompt):
    # Create output directories if they don't exist
    os.makedirs(f"../data/delegate/{trial}", exist_ok=True)
    os.makedirs(f"../data/trustee/{trial}", exist_ok=True)
    
    for i in range(start_index, len(policies)):
        policy = policies.iloc[i].statement
        
        # Create empty files for this policy if they don't exist
        delegate_file = f"../data/delegate/{trial}/d_policy_{i+1}_votes.jsonl"
        trustee_file = f"../data/trustee/{trial}/t_policy_{i+1}_votes.jsonl"
        
        if not os.path.exists(delegate_file):
            open(delegate_file, 'w').close()
        if not os.path.exists(trustee_file):
            open(trustee_file, 'w').close()
        
        for index, row in written_profiles.iterrows():
            bio_dict = row.to_dict()
            
            # Get delegate vote
            response_delegate = run_claude(delegate_prompt, bio_dict['Profile'], policy)
            processed_response_delegate = process_vote_response(response_delegate, bio_dict['ID'])
            
            # Get trustee vote
            response_trustee = run_claude(trustee_prompt, bio_dict['Profile'], policy)
            processed_response_trustee = process_vote_response(response_trustee, bio_dict['ID'])
            
            print(f"Voted for {bio_dict['ID']}")
            
            # Save votes
            with open(delegate_file, "a") as file:
                file.write(processed_response_delegate + "\n")
            with open(trustee_file, "a") as file:
                file.write(processed_response_trustee + "\n")

def main():
    args = parse_args()
    
    # Load prompts
    prompts = load_prompts()
    if str(args.prompt_num) not in prompts:
        raise ValueError(f"Prompt number {args.prompt_num} not found in prompts.json")
    
    prompt_set = prompts[str(args.prompt_num)]
    delegate_prompt = prompt_set['delegate']
    trustee_prompt = prompt_set['trustee']
    
    # Load data
    global written_profiles, policies
    written_profiles = pd.read_json("gpt-4o_written_profiles.jsonl", encoding='cp1252', lines=True)
    policies = pd.read_json("../self_selected_policies.jsonl", lines=True)
    
    # Run voting simulation
    vote_on_policies(args.trial, args.start_index, delegate_prompt, trustee_prompt)

if __name__ == "__main__":
    main() 