import re

def run_code(new_code: str):
    """
    Extracts and executes Python code blocks from a string using regex.
    Matches blocks like: ```python\n<code>\n```
    """
    pattern = re.compile(r'```python\s*\n(.*?)```', re.DOTALL)
    code_blocks = pattern.findall(new_code)

    if not code_blocks:
        print("⚠️ No Python code blocks found.")
        return

    for code in code_blocks:
        exec(code, globals())
