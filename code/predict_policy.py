#!/usr/bin/env python3
"""
Single Policy Prediction Runner
Command-line script to predict utilities for a single policy.
Designed for parallel execution across multiple policies.
"""

import argparse
import json
import pandas as pd
import os
import anthropic
import sys
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def load_data(prompt_file='prompts_long_short.json', policy_file='../self_selected_policies_new.jsonl'):
    """Load prompts, profiles, and policies data."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading data...")

    # Load prompts from specified file
    prompt_path = f'../{prompt_file}'
    with open(prompt_path, 'r') as f:
        prompts = json.load(f)

    # Load user profiles
    written_profiles = pd.read_json("gpt-4o_written_profiles.jsonl", encoding='cp1252', lines=True)

    # Load policies
    print(policy_file)
    policies = pd.read_json(policy_file, lines=True)
    print(policies.head())
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded {len(prompts)} prompt sets from {prompt_file}, {len(written_profiles)} profiles, {len(policies)} policies")

    return prompts, written_profiles, policies

def run_claude(prompt, profile, policy):
    """Execute Claude API call."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    messages = [
        {
            "role": "user",
            "content": f"{prompt.format(bio=profile)}\n\nPolicy proposal: {policy}"
        }
    ]

    system = prompt.format(bio=profile)

    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=1024,
        messages=messages,
        system=system,
        temperature=0.0
    )

    return response.content[0].text

def run_gpt(prompt, profile, policy):
    """Execute GPT API call with full prompt in both system and user messages."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Format the full prompt with bio and policy
    full_prompt = f"{prompt.format(bio=profile)}\n\nPolicy proposal: {policy}"

    messages = [
        {
            "role": "system",
            "content": full_prompt
        },
        {
            "role": "user",
            "content": full_prompt
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.0
    )

    return response.choices[0].message.content

def get_llm_response(prompt, profile, policy, model_name):
    """Route to appropriate LLM while using same prompts."""
    if model_name == "gpt-4o":
        return run_gpt(prompt, profile, policy)
    elif model_name.startswith("claude"):
        return run_claude(prompt, profile, policy)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def predict_policy(policy_index, prompt_type, policy_file, model_name, prompt_num, n_users, prompts, written_profiles, policies):
    """
    Predict utilities for all users on a single policy.

    Args:
        policy_index (int): Index of policy to test (0-based)
        prompt_type (str): "delegate_ls" or "trustee_ls"
        policy_file (str): Policy file to use
        model_name (str): Model name for directory structure
        prompt_num (int): Prompt number for directory structure
        n_users (int or None): Number of users to process (None for all)
        prompts (dict): Loaded prompts data
        written_profiles (DataFrame): User profiles
        policies (DataFrame): Policy statements
    """
    start_time = datetime.now()
    print(f"[{start_time.strftime('%H:%M:%S')}] Starting policy {policy_index + 1} prediction")
    print("=" * 80)

    # Get policy and prompt
    policy = policies.iloc[policy_index].statement
    prompt = prompts[str(prompt_num)][prompt_type]

    # Create output directory
    policy_file_clean = policy_file.split("/")[1].split(".")[0]
    print(policy_file_clean)
    #raise ValueError("Stop here")
    output_dir = f"../data/{prompt_type}/{model_name}/{policy_file_clean}/prompt-{prompt_num}"
    os.makedirs(output_dir, exist_ok=True)


    # Output file path
    prefix = "t_" if "trustee" in prompt_type else "d_"
    output_file = f"{output_dir}/{prefix}policy_{policy_index + 1}_votes.jsonl"

    # Check if file already exists
    if os.path.exists(output_file):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠ Output file already exists: {output_file}")
        response = input("Overwrite? (y/N): ").strip().lower()
        if response != 'y':
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Skipping policy {policy_index + 1}")
            return

    # Determine how many users to process
    users_to_process = written_profiles.head(n_users) if n_users else written_profiles

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Policy {policy_index + 1}: {policy}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Prompt type: {prompt_type}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Model: {model_name}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Output: {output_file}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing {len(users_to_process)} users...")
    print("=" * 80)

    # Open file for immediate writing
    results = []
    errors = []

    with open(output_file, 'w') as f:
        # Process each user
        for idx, row in users_to_process.iterrows():
            user_id = row['ID']
            profile = row['Profile']

            try:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [{len(results)+1:3d}/{len(users_to_process)}] Processing user {user_id}...", end=" ")

                # Get LLM response
                response = get_llm_response(prompt, profile, policy, model_name)

                # Process response
                clean_response = response.replace("\n", " ")
                try:
                    # Try to extract JSON
                    json_start = clean_response.find('{')
                    json_end = clean_response.rfind('}') + 1
                    json_str = clean_response[json_start:json_end]

                    # Parse and add ID
                    parsed = json.loads(json_str)
                    parsed['id'] = user_id

                    # Write immediately to file
                    f.write(json.dumps(parsed) + "\n")
                    f.flush()  # Ensure it's written to disk

                    results.append(parsed)
                    print("✓ Success")

                except (json.JSONDecodeError, ValueError) as e:
                    # If JSON parsing fails, create basic structure
                    result = {
                        'id': user_id,
                        'response': clean_response
                    }

                    # Write immediately to file
                    f.write(json.dumps(result) + "\n")
                    f.flush()  # Ensure it's written to disk

                    results.append(result)
                    print(f"⚠ JSON parse failed, saved raw response")

                # Progress summary every 25 users
                if len(results) % 25 == 0:
                    elapsed = datetime.now() - start_time
                    rate = len(results) / elapsed.total_seconds() * 60  # users per minute
                    remaining = len(users_to_process) - len(results)
                    eta_minutes = remaining / (rate / 60) if rate > 0 else 0

                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] --- Progress: {len(results)}/{len(users_to_process)} ({len(results)/len(users_to_process)*100:.1f}%) ---")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Rate: {rate:.1f} users/min, ETA: {eta_minutes:.1f} min")
                    if errors:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Errors so far: {len(errors)} users")
                    print()

            except Exception as e:
                error_msg = f"User {user_id}: {str(e)}"
                errors.append(error_msg)
                print(f"✗ ERROR: {e}")
                continue

    # Final summary
    end_time = datetime.now()
    elapsed = end_time - start_time

    print("=" * 80)
    print(f"[{end_time.strftime('%H:%M:%S')}] ✅ POLICY {policy_index + 1} COMPLETED!")
    print(f"[{end_time.strftime('%H:%M:%S')}] Total time: {elapsed}")
    print(f"[{end_time.strftime('%H:%M:%S')}] Successfully processed: {len(results)} users")
    print(f"[{end_time.strftime('%H:%M:%S')}] Errors encountered: {len(errors)} users")
    print(f"[{end_time.strftime('%H:%M:%S')}] Success rate: {len(results)/(len(results)+len(errors))*100:.1f}%")
    print(f"[{end_time.strftime('%H:%M:%S')}] Output file: {output_file}")

    if errors:
        print(f"\n[{end_time.strftime('%H:%M:%S')}] ⚠ Error summary:")
        for error in errors:
            print(f"[{end_time.strftime('%H:%M:%S')}]   - {error}")

    return len(results), len(errors)

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Predict utilities for a single policy')

    parser.add_argument('--policy', type=int, required=True, default=0,
                        help='Policy index to process (0-based)')
    parser.add_argument('--policy-file', type=str, default='../self_selected_policies_new.jsonl',
                        help='Policy file to use')
    parser.add_argument('--prompt-type', type=str, default='trustee_ls',
                        choices=['delegate_ls', 'trustee_ls', 'delegate_lsd', 'trustee_lsd'],
                        help='Type of prompt to use')
    parser.add_argument('--model', type=str, default='claude-3-sonnet-v2',
                        help='Model name (claude-3-sonnet-v2 or gpt-4o)')
    parser.add_argument('--prompt-num', type=int, default=0,
                        help='Prompt number to use')
    parser.add_argument('--prompt-file', type=str, default='prompts_long_short.json',
                        choices=['prompts_long_short.json', 'prompts_long_short_discount.json'],
                        help='Prompt file to use')
    parser.add_argument('--n-users', type=int, default=None,
                        help='Number of users to process (default: all)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    # Validate arguments
    if args.policy < 0:
        print("Error: Policy index must be non-negative")
        sys.exit(1)

    if args.n_users is not None and args.n_users <= 0:
        print("Error: Number of users must be positive")
        sys.exit(1)

    # Validate API keys based on model
    if args.model == "gpt-4o":
        if "OPENAI_API_KEY" not in os.environ:
            print("Error: OPENAI_API_KEY environment variable not set for GPT-4o")
            sys.exit(1)
    elif args.model.startswith("claude"):
        if "ANTHROPIC_API_KEY" not in os.environ:
            print("Error: ANTHROPIC_API_KEY environment variable not set for Claude")
            sys.exit(1)

    try:
        # Load data
        prompts, written_profiles, policies = load_data(args.prompt_file, args.policy_file)

        # Validate policy index
        if args.policy >= len(policies):
            print(f"Error: Policy index {args.policy} out of range (0-{len(policies)-1})")
            sys.exit(1)

        # Validate prompt
        if str(args.prompt_num) not in prompts:
            print(f"Error: Prompt number {args.prompt_num} not found in prompts")
            sys.exit(1)

        if args.prompt_type not in prompts[str(args.prompt_num)]:
            print(f"Error: Prompt type '{args.prompt_type}' not found for prompt {args.prompt_num}")
            sys.exit(1)

        # Run prediction
        success_count, error_count = predict_policy(
            args.policy, args.prompt_type, args.policy_file, args.model,
            args.prompt_num, args.n_users,
            prompts, written_profiles, policies
        )

        # Exit with appropriate code
        if error_count == 0:
            sys.exit(0)  # Success
        elif success_count > 0:
            sys.exit(1)  # Partial success
        else:
            sys.exit(2)  # Complete failure

    except KeyboardInterrupt:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ⚠ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Fatal error: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()