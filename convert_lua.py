
import json

with open('./he_retrieved_examples_mpnet_10_mbpp_py_by_ours_for_lua.json', 'r') as f:
    data = json.load(f)
    
    for prob in data:
        cur_prob = data[prob]
        for example in cur_prob:
            prompt = example['prompt']
            
            # if prompt.startswith("#include"):
            #     lines = prompt.strip().split('\n')
            #     comment_lines = [line.strip() for line in lines if line.strip().startswith('//')]
            #     prompt = comment_lines[-1].replace("// ", "")
            
            code = example['code']
            
            new_prompt = f'    """{prompt}\n    """'
            
            lines = code.split('\n')
            
            new_lines = []
            prompt_idx = 0
            
            for idx, line in enumerate(lines):
                if line.startswith('def'):
                    line += "\n" +new_prompt
                    prompt_idx = idx
                new_lines.append(line)
            
            new_prompt = "\n".join(new_lines[:prompt_idx+1])
            new_code = "\n".join(new_lines)
            
            print(prompt)
            example['prompt'] = new_prompt
            # example['code'] = new_code
            # example['final_plan'] = ""
            # example['requirements'] = ""
    
    with open('data/he_retrieved_examples_mpnet_10_mbpp_py_by_ours_for_lua_formatted.json', 'w') as f:
        json.dump(data, f, indent=4)