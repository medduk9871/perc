
import json

with open('./he_retrieved_examples_mpnet_10_mbpp_r_prob2prob.json', 'r') as f:
    data = json.load(f)
    
    for prob in data:
        cur_prob = data[prob]
        for example in cur_prob:
            prompt = example['prompt']
            code = example['code']
            
            new_prompt = f'# {prompt}\n'
            
            lines = code.split('\n')
            
            new_lines = []
            prompt_idx = 0
            
            for idx, line in enumerate(lines):
                if '<- function' in line:
                    line = new_prompt + line + "\n"
                    prompt_idx = idx
                new_lines.append(line)
            
            new_prompt = "\n".join(new_lines[:prompt_idx+1])
            new_code = "\n".join(new_lines)
            
            print(new_prompt)
            example['prompt'] = new_prompt
            example['code'] = code
            example['final_plan'] = ""
            example['requirements'] = ""
    
    with open('data/he_retrieved_examples_mpnet_10_mbpp_r_prob2prob_formatted.json', 'w') as f:
        json.dump(data, f, indent=4)