#!/usr/bin/env python3
"""
Script to fix malformed JSON entries in trustee_lsd data files.

The script identifies entries with 'response' field containing markdown text
and uses GPT-4o to extract the properly formatted JSON structure.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime
import shutil

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Run: pip install openai")
    sys.exit(1)

# Configuration
PROMPT_DIR = "../data/trustee_lsd/claude-3-sonnet-v2/self_selected_policies_new/prompt-2"
BACKUP_SUFFIX = ".backup"
DRY_RUN = False  # Set to True to see what would be changed without making changes

# OpenAI API setup
client = OpenAI()  # Expects OPENAI_API_KEY environment variable

EXTRACTION_PROMPT = """You are a data cleaning assistant. I have a malformed JSON entry that contains the correct data embedded in markdown format.

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
            # Remove first and last lines if they're markdown delimiters
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            fixed_json_text = "\n".join(lines)

        # Parse the JSON
        fixed_entry = json.loads(fixed_json_text)

        # Validate the fixed entry
        if is_valid_format(fixed_entry):
            return fixed_entry
        else:
            print(f"Warning: GPT returned invalid format for id {malformed_entry.get('id', 'unknown')}")
            return None

    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse GPT response for id {malformed_entry.get('id', 'unknown')}: {e}")
        return None
    except Exception as e:
        print(f"Error: Failed to fix entry with id {malformed_entry.get('id', 'unknown')}: {e}")
        return None


def process_file(file_path: Path, dry_run: bool = False) -> Tuple[int, int, int]:
    """
    Process a single JSONL file and fix malformed entries.

    Returns:
        Tuple of (total_entries, malformed_entries, fixed_entries)
    """
    print(f"\nProcessing: {file_path.name}")

    # Read all entries
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                entries.append((line_num, entry))
            except json.JSONDecodeError as e:
                print(f"  Warning: Line {line_num} is not valid JSON: {e}")
                entries.append((line_num, None))

    total_entries = len(entries)
    malformed_count = sum(1 for _, entry in entries if entry and is_malformed(entry))

    print(f"  Total entries: {total_entries}")
    print(f"  Malformed entries: {malformed_count}")

    if malformed_count == 0:
        print("  No malformed entries found. Skipping.")
        return total_entries, 0, 0

    if dry_run:
        print(f"  [DRY RUN] Would fix {malformed_count} entries")
        return total_entries, malformed_count, 0

    # Create backup
    backup_path = file_path.with_suffix(file_path.suffix + BACKUP_SUFFIX)
    shutil.copy2(file_path, backup_path)
    print(f"  Backup created: {backup_path.name}")

    # Fix malformed entries
    fixed_entries = []
    fixed_count = 0
    failed_count = 0

    for line_num, entry in entries:
        if entry is None:
            # Keep invalid lines as-is (shouldn't happen if we got here)
            continue

        if is_malformed(entry):
            print(f"  Fixing entry {entry.get('id', 'unknown')} (line {line_num})...", end=" ")
            fixed_entry = fix_entry_with_gpt(entry)

            if fixed_entry:
                fixed_entries.append(fixed_entry)
                fixed_count += 1
                print("✓")
            else:
                # Keep the original if fixing failed
                fixed_entries.append(entry)
                failed_count += 1
                print("✗ (keeping original)")

            # Rate limiting to avoid API throttling
            time.sleep(0.5)
        else:
            # Keep valid entries as-is
            fixed_entries.append(entry)

    # Write the fixed file
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in fixed_entries:
            f.write(json.dumps(entry) + '\n')

    print(f"  Fixed: {fixed_count}/{malformed_count} entries")
    if failed_count > 0:
        print(f"  Failed: {failed_count} entries (kept original)")

    return total_entries, malformed_count, fixed_count


def main():
    """Main function to process all files in the prompt-2 directory."""
    print("=" * 70)
    print("Malformed JSON Entry Fixer for trustee_lsd data")
    print("=" * 70)
    print(f"Mode: {'DRY RUN' if DRY_RUN else 'LIVE RUN'}")
    print(f"Directory: {PROMPT_DIR}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Check if directory exists
    prompt_dir = Path(PROMPT_DIR)
    if not prompt_dir.exists():
        print(f"Error: Directory not found: {PROMPT_DIR}")
        sys.exit(1)

    # Get all JSONL files
    jsonl_files = sorted(prompt_dir.glob("t_policy_*_votes.jsonl"))

    if not jsonl_files:
        print("Error: No matching files found")
        sys.exit(1)

    print(f"Found {len(jsonl_files)} files to process")

    # Confirm before proceeding
    if not DRY_RUN:
        response = input("\nThis will modify files. Backups will be created. Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted by user")
            sys.exit(0)

    # Process all files
    total_entries = 0
    total_malformed = 0
    total_fixed = 0
    start_time = time.time()

    for file_path in jsonl_files:
        entries, malformed, fixed = process_file(file_path, dry_run=DRY_RUN)
        total_entries += entries
        total_malformed += malformed
        total_fixed += fixed

    # Summary
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Files processed: {len(jsonl_files)}")
    print(f"Total entries: {total_entries}")
    print(f"Malformed entries found: {total_malformed}")
    print(f"Successfully fixed: {total_fixed}")
    if total_malformed > 0:
        print(f"Success rate: {total_fixed/total_malformed*100:.1f}%")
    print(f"Time elapsed: {elapsed_time:.1f} seconds")

    if not DRY_RUN and total_fixed > 0:
        print(f"\nBackup files created with suffix: {BACKUP_SUFFIX}")
        print("To restore backups if needed, use the restore_backups() function")

    print("=" * 70)


def restore_backups():
    """Restore all backup files (in case something went wrong)."""
    prompt_dir = Path(PROMPT_DIR)
    backup_files = list(prompt_dir.glob(f"*{BACKUP_SUFFIX}"))

    if not backup_files:
        print("No backup files found")
        return

    print(f"Found {len(backup_files)} backup files")
    response = input("Restore all backups? This will overwrite current files (yes/no): ")

    if response.lower() != 'yes':
        print("Aborted")
        return

    for backup_path in backup_files:
        original_path = backup_path.with_suffix('')
        shutil.copy2(backup_path, original_path)
        print(f"Restored: {original_path.name}")

    print("All backups restored")


if __name__ == "__main__":
    main()