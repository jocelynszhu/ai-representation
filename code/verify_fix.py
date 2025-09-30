#!/usr/bin/env python3
"""Quick verification of the fix quality by inspecting a fixed entry."""

import json
from openai import OpenAI

client = OpenAI()

# Get one malformed entry from the file
with open("../data/trustee_lsd/claude-3-sonnet-v2/self_selected_policies_new/prompt-2/t_policy_1_votes.jsonl", 'r') as f:
    for line in f:
        entry = json.loads(line.strip())
        if "response" in entry and "yes" not in entry:
            malformed = entry
            break

print("ORIGINAL MALFORMED ENTRY:")
print("=" * 70)
print(f"ID: {malformed['id']}")
print(f"Has 'response' field: Yes")
print(f"Response preview: {malformed['response'][:200]}...")
print()

# Fix it
EXTRACTION_PROMPT = """You are a data cleaning assistant. Extract ONLY the JSON object from this malformed entry.

Return format:
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

Malformed entry:
"""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a data extraction expert."},
        {"role": "user", "content": EXTRACTION_PROMPT + json.dumps(malformed, indent=2)}
    ],
    temperature=0,
    max_tokens=4000
)

fixed_text = response.choices[0].message.content.strip()
if fixed_text.startswith("```"):
    lines = fixed_text.split("\n")
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    fixed_text = "\n".join(lines)

fixed = json.loads(fixed_text)

print("FIXED ENTRY:")
print("=" * 70)
print(json.dumps(fixed, indent=2))
print()
print("VALIDATION:")
print("=" * 70)
print(f"Has 'id': {' id' in fixed}")
print(f"Has 'yes': {'yes' in fixed}")
print(f"Has 'no': {'no' in fixed}")
print(f"All time periods present: ", end="")

required_periods = ["0-5 years", "5-10 years", "10-15 years", "15-20 years", "20-25 years", "25-30 years"]
all_present = all(
    period in fixed['yes'] and period in fixed['no']
    for period in required_periods
)
print("✓" if all_present else "✗")

print(f"\nSample yes/0-5 years: score={fixed['yes']['0-5 years']['score']}")
print(f"Sample no/0-5 years: score={fixed['no']['0-5 years']['score']}")