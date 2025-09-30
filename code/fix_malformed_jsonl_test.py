#!/usr/bin/env python3
"""
Test script to fix malformed JSON entries in a single file.
This is a test version before running on all 30 files.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional
import time
import shutil

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Run: pip install openai")
    sys.exit(1)

# Configuration
TEST_FILE = "../data/trustee_lsd/claude-3-sonnet-v2/self_selected_policies_new/prompt-2/t_policy_1_votes.jsonl"
BACKUP_SUFFIX = ".test_backup"

# OpenAI API setup
client = OpenAI()  # Expects OPENAI_API_KEY environment variable

EXTRACTION_PROMPT = """You are a data cleaning assistant. I have a malformed JSON entry that contains the correct data embedded in markdown format.
p
Your task: Extract ONLY the JSON object from the markdown text and return it as valid JSON.

The correct format should be:
{
  "yes": {
    "0-5 years": {"rationale": "...", "score": <number>},
    "5-10 years": {"rationale": "...", "score": <number>},
    "10-15 years": {"rationale": "...", "score": <number>},
    "15-20 years": {"rationale": "...", "score": <number>},
    "20-25 years": {"rationale": "...", "score": <number>},
    "25-30 years": {"rationale": "...", "score": <number>}
  },
  "no": {
    "0-5 years": {"rationale": "...", "score": <number>},
    "5-10 years": {"rationale": "...", "score": <number>},
    "10-15 years": {"rationale": "...", "score": <number>},
    "15-20 years": {"rationale": "...", "score": <number>},
    "20-25 years": {"rationale": "...", "score": <number>},
    "25-30 years": {"rationale": "...", "score": <number>}
  },
  "id": <number>
}

Rules:
1. Extract the complete JSON object from any markdown code blocks
2. Preserve the "id" field from the original entry
3. Return ONLY valid JSON, no markdown, no explanations
4. If the JSON is truncated in the markdown, try to complete it based on the pattern
5. Ensure all 6 time periods are present for both "yes" and "no"

Here is the malformed entry:
"""


def is_malformed(entry: Dict) -> bool:
    """Check if an entry has the malformed format with 'response' field."""
    return "response" in entry and "yes" not in entry


def is_valid_format(entry: Dict) -> bool:
    """Check if an entry has the correct format."""
    if "id" not in entry:
        return False
    if "yes" not in entry or "no" not in entry:
        return False

    required_periods = ["0-5 years", "5-10 years", "10-15 years",
                       "15-20 years", "20-25 years", "25-30 years"]

    for vote_type in ["yes", "no"]:
        if not isinstance(entry[vote_type], dict):
            return False
        for period in required_periods:
            if period not in entry[vote_type]:
                return False
            period_data = entry[vote_type][period]
            if not isinstance(period_data, dict):
                return False
            if "rationale" not in period_data or "score" not in period_data:
                return False

    return True


def fix_entry_with_gpt(malformed_entry: Dict) -> Optional[Dict]:
    """Use GPT-4o to extract the correct JSON from a malformed entry."""
    try:
        entry_text = json.dumps(malformed_entry, indent=2)

        print(f"    Sending to GPT-4o...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a data extraction expert. Extract valid JSON from malformed data."},
                {"role": "user", "content": EXTRACTION_PROMPT + entry_text}
            ],
            temperature=0,
            max_tokens=4000
        )

        # Extract the response
        fixed_json_text = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if fixed_json_text.startswith("```"):
            lines = fixed_json_text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            fixed_json_text = "\n".join(lines)

        # Parse the JSON
        fixed_entry = json.loads(fixed_json_text)

        # Validate the fixed entry
        if is_valid_format(fixed_entry):
            print(f"    ✓ Successfully fixed entry")
            return fixed_entry
        else:
            print(f"    ✗ GPT returned invalid format")
            return None

    except json.JSONDecodeError as e:
        print(f"    ✗ Failed to parse GPT response: {e}")
        return None
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return None


def main():
    """Test the fix on a single file with just the first few malformed entries."""
    print("=" * 70)
    print("TEST: Malformed JSON Entry Fixer")
    print("=" * 70)
    print(f"Test file: {TEST_FILE}")
    print("=" * 70)

    file_path = Path(TEST_FILE)
    if not file_path.exists():
        print(f"Error: File not found: {TEST_FILE}")
        sys.exit(1)

    # Read all entries
    print("\n1. Reading file...")
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                entries.append((line_num, entry))
            except json.JSONDecodeError as e:
                print(f"  Warning: Line {line_num} is not valid JSON")
                entries.append((line_num, None))

    total = len(entries)
    malformed = [e for _, e in entries if e and is_malformed(e)]
    valid = [e for _, e in entries if e and is_valid_format(e)]

    print(f"  Total entries: {total}")
    print(f"  Valid entries: {len(valid)}")
    print(f"  Malformed entries: {len(malformed)}")

    if not malformed:
        print("\nNo malformed entries found!")
        return

    # Test fixing just the first 3 malformed entries
    print(f"\n2. Testing fix on first 3 malformed entries...")
    test_limit = min(3, len(malformed))

    fixed_count = 0
    for i, entry in enumerate(malformed[:test_limit], 1):
        print(f"\n  Entry {i}/{test_limit} (id={entry.get('id', 'unknown')}):")
        fixed = fix_entry_with_gpt(entry)
        if fixed:
            fixed_count += 1
            # Show a preview
            print(f"    Preview: yes/0-5 years score = {fixed['yes']['0-5 years']['score']}")

        # Small delay to be nice to the API
        if i < test_limit:
            time.sleep(1)

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"Tested: {test_limit} entries")
    print(f"Successfully fixed: {fixed_count}/{test_limit}")
    print(f"Success rate: {fixed_count/test_limit*100:.0f}%")
    print("\nIf this looks good, you can run the full script on all files.")
    print("=" * 70)


if __name__ == "__main__":
    main()