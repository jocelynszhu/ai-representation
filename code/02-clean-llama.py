#%%
#%%
import json
import pandas as pd 
import os
# parse_and_write_jsonl('path/to/input.txt', 'path/to/output.jsonl')(drop=True, inplace=True)
#df
#%%
num_policies = 20
model = 'claude-3-sonnet-v2'
delegate_or_trustee = ['delegate', 'trustee']
prompt = 'prompt-3'
for policy in range(num_policies):
    print(f'Processing policy {policy+1} of {num_policies}')
    for do_t in delegate_or_trustee:
        print(f'Processing {do_t} for policy {policy+1}')
        input_filename = f'../data/{do_t}/{model}-old/{prompt}/{do_t[0]}_policy_{policy+1}_votes.jsonl'
        print(input_filename)
        # Create output directory if it doesn't exist
        os.makedirs(f'../data/{do_t}/{model}/{prompt}', exist_ok=True)
        output_filename = f'../data/{do_t}/{model}/{prompt}/{do_t[0]}_policy_{policy+1}_votes.jsonl'
        with open(input_filename, 'r') as f:
            lines = f.readlines()

        with open(output_filename, 'w') as out_f:
            i = 0
            while i < len(lines):
                try:
                    # First try to read a single line
                    single_line = lines[i].strip()
                    if single_line:  # Only process non-empty lines
                        # Check if line only contains an ID
                        if single_line.startswith('"id":'):
                            # Extract just the ID number
                            id_num = single_line.split(':')[1].strip().rstrip(',')
                            # Format with default values 
                            formatted_obj = {
                                "id": int(id_num),
                                "reason": "NA",
                                "vote": "NA"
                            }
                            out_f.write(json.dumps(formatted_obj) + '\n')
                            i += 1
                            continue
                        if single_line.startswith('{"id":') and single_line.endswith('}'):
                            try:
                                obj = json.loads(single_line)
                  
                            except json.JSONDecodeError:
                                pass

                        try:
                            obj = json.loads(single_line)
                            out_f.write(json.dumps(obj) + '\n')
                            i += 1
                            continue  # Skip to next iteration if successful
                        except json.JSONDecodeError:
                            # If single line parsing fails, try 4-line chunk
                            pass
                    
                    # Try 4-line chunk if single line parsing failed or line was empty
                    chunk = lines[i:i+4]
                    if len(chunk) == 4:
                        #print(chunk)
                        raw_json = ''.join(chunk).strip()
                        obj = json.loads(raw_json)
                        out_f.write(json.dumps(obj) + '\n')
                        i += 4
                    else:
                        i += 1  # Move to next line if both attempts fail
                except Exception as e:
                    print(f"Error parsing JSON object starting at line {i+1}:")
                    print(''.join(chunk) if len(chunk) == 4 else lines[i])
                    raise e  # Stop so you can manually fix the input file


#%%

#%%