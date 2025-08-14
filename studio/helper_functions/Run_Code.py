import re
from classes import State

def run_code(state:State, new_code: str):
    """
    Extracts and executes Python code blocks from a string using regex.
    Matches blocks like: ```python\n<code>\n```
    """
    print(f"This is the new code \n\n {new_code} \n \n")
    if state['code_type'] == "vegalite": pattern = re.compile(r'```json\s*\n(.*?)```', re.DOTALL)
    else: pattern = re.compile(r'```python\s*\n(.*?)```', re.DOTALL)
    code_blocks = pattern.findall(new_code)

    if not code_blocks:
        if state['code_type'] == "vegalite": print("⚠️ No HTML code blocks found.")
        else: print("⚠️ No Python code blocks found.")
        return

    for code in code_blocks:
        exec(code, globals())
