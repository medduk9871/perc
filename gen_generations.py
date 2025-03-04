import json
import datasets
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--lang', type=str, required=True)
parser.add_argument('-d', '--dir', type=str, required=True)
args = parser.parse_args()

CPP_PRED_TEMPLATE = """\
#include<assert.h>
#include<bits/stdc++.h>
"""
JAVA_PRED_TEMPLATE = """\

"""
PY_PRED_TEMPLATE = """\
import sys
import math
import re
import numpy
import numpy as np
from typing import *

"""
LUA_PRED_TEMPLATE = """\

"""
RUBY_PRED_TEMPLATE = """\

"""
LANG = args.lang or "r"
RESULT_DIR = args.dir or f"./results/humaneval-{LANG}-gpt35-rag-3shot-ours-formatted"

def remove_code_block(code):
    # Split the Lua code into lines
    
    # langs = ['rb', 'r', 'py', 'python', 'lua', 'ruby', 'RB', 'R', 'PY', 'PYTHON', 'LUA', 'RUBY']
    if LANG == "rb":
        langs = ['rb', 'ruby', 'RUBY', 'RB']
    elif LANG == 'r':
        langs = ['r', 'R']
    elif LANG == 'lua':
        langs = ['lua']
    elif LANG == 'py':
        langs == ['python', 'PYTHON', 'py', 'PY']
    
    for lang in langs:
        if f'```{lang}' in code:
            code = code.split(f'```{lang}')[1]
            lines = code.strip().split('\n')
            
            # Filter out lines containing the code block delimiters
            code = '\n'.join(line for line in lines if not line.strip().startswith(f'```{lang}') and not line.strip().endswith('```'))
        
    return code

with open(f"{RESULT_DIR}/result.json", 'r') as f:
    data = json.load(f)
    
    code = []
    
    problems = datasets.load_dataset("nuprl/MultiPL-E", f"humaneval-{LANG}", revision='5d2abbb8ced9a0e37db985c47d24c24f45a16655')
    
    for prob in problems['test']:
        name = "/".join(prob['name'].split('_')[:2])
        target_func_name = prob['name'].split('_')[2]

        for item in data:
            if name == item['id']:
                prefixed_list = []
                for cur_code in item['code']:
                    cur_code = remove_code_block(cur_code)
                    import re
                    
                    if LANG == "java":
                        def remove_main_method(java_code):
                            # Define a regex pattern to match everything from "public static void main(String[] args)" to the end of the method
                            pattern = re.compile(r'public\s+static\s+void\s+main\s*\(.*?\)\s*{.*?}', re.DOTALL)
                            # Substitute the matched pattern with an empty string
                            modified_code = re.sub(pattern, '', java_code)
                            return modified_code.strip()
                        
                        # first_brace_idx = cur_code.find('{')
                        # last_sig_idx = item["prompt"].rfind('public')
                        # new_code = item["prompt"][:last_sig_idx]+ cur_code[first_brace_idx+1:]
                        new_code = remove_main_method(cur_code)
                        last_brace_idx = new_code.rfind('}')
                        new_code = new_code[:last_brace_idx]
                        last_brace_idx = new_code.rfind('}')
                        new_code = new_code[:last_brace_idx]
                        
                        new_code = JAVA_PRED_TEMPLATE + new_code
                        
                        print(new_code)
                        
                        # if "94" in item['id']:
                        #     print(item['id'])
                        #     print(new_code)
                    
                    elif LANG == "cpp":
            
                        def remove_main_method(cpp_code):
                            # Define a regex pattern to match everything from "int main()" or "void main()" to the end of the method
                            pattern = re.compile(r'(int\s+main\s*\(.*?\)\s*{.*|void\s+main\s*\(.*?\)\s*{.*)', re.DOTALL)
                            # Substitute the matched pattern with an empty string
                            modified_code = re.sub(pattern, '', cpp_code)
                            return modified_code.strip()
                        
                        new_code = cur_code
                        
                        new_code = remove_main_method(new_code)
                        last_brace_idx = new_code.rfind('}')
                        
                        new_code = new_code[:last_brace_idx]
                        
                        if "assert.h" not in new_code:
                            new_code = CPP_PRED_TEMPLATE + new_code
                        print(new_code)
                    elif LANG == "py":
                        new_code = PY_PRED_TEMPLATE + cur_code
                        print(new_code)
                    elif LANG == "lua":
                        new_code = LUA_PRED_TEMPLATE + cur_code
                        
                        if "```lua" in new_code:
                            new_code = remove_code_block(new_code)
                        
                        if '-- Test' in new_code:
                            new_code = new_code.split('-- Test')[0]
                            
                        if '-- test' in new_code:
                            new_code = new_code.split('-- test')[0]
                        
                        import re

                        def remove_code_after_last_end(code):
                            # Regular expression to find the last occurrence of 'end'
                            last_end_pattern = re.compile(r'^(.*end\b)', re.DOTALL)

                            # Search for the last occurrence of 'end' in the code
                            match = last_end_pattern.search(code)
                            
                            if match:
                                return match.group(1)
                            else:
                                return code
                        new_code = remove_code_after_last_end(new_code)
                        print(new_code)
                    elif LANG == "rb":
                        new_code = RUBY_PRED_TEMPLATE + cur_code
                        
                        if "```rb" in new_code:
                            new_code = remove_code_block(new_code)
                        if '# Test' in new_code:
                            new_code = new_code.split('# Test')[0]
                            
                        if '# test' in new_code:
                            new_code = new_code.split('# test')[0]
                            
                        if '# Example' in new_code:
                            new_code = new_code.split('# Example')[0]
                        
                        import re

                        def remove_code_after_last_end(code):
                            # Regular expression to find the last occurrence of 'end'
                            last_end_pattern = re.compile(r'^(.*end\b)', re.DOTALL)

                            # Search for the last occurrence of 'end' in the code
                            match = last_end_pattern.search(code)
                            
                            if match:
                                return match.group(1)
                            else:
                                return code
                        new_code = remove_code_after_last_end(new_code)
                        if 'end' not in new_code and 'lambda' not in new_code:
                            new_code += "\nend\n"
                        
                        lines = new_code.split('\n')
                        new_lines = []
                        
                        for line in lines:
                            if line.endswith(':'):
                                line = line[:-1]
                            new_lines.append(line)
                        
                        new_code = '\n'.join(new_lines)
                        print(new_code)
                    
                    elif LANG == "r":
                        new_code = RUBY_PRED_TEMPLATE + cur_code
                        
                        if "```r" in new_code:
                            new_code = remove_code_block(new_code)
                        if '# Test' in new_code:
                            new_code = new_code.split('# Test')[0]
                        if '# Example' in new_code:
                            new_code = new_code.split('# Example')[0]
                            
                        
                        if not new_code.endswith('}'):
                            # Traverse the lines in reverse order and find the first '}'
                            lines = new_code.split('\n')
                            for i in range(len(lines) - 1, -1, -1):
                                if '}' in lines[i]:
                                    # Retain all lines up to and including this line
                                    lines = lines[:i + 1]
                                    break
                            new_code = '\n'.join(lines)
                        print(new_code)
                    
                    prefixed_list.append(new_code)

                code.append(prefixed_list)
                break
    
with open(f"{RESULT_DIR}/generations.json", 'w') as f:
    json.dump(code, f, indent=4)