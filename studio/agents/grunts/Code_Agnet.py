import re
from langgraph.graph import END
from langgraph.types import Command
from classes import State
from typing import Literal
from helper_functions import generate_pdf_report
from langchain_experimental.utilities import PythonREPL

def code_agent(state:State) -> Command[Literal[END, 'vis_agent']]:
    repl = PythonREPL()
    code = state['messages'][-1].content
    try:
        # Regex to extract content inside triple single quotes
        #py_code = re.findall(r"```python\n(.*?)```", code, re.DOTALL)
        #print(f'\n\n this is the code {py_code[0]} \n\n')
        #result = repl.run(py_code[0])
        #print(f"\n\n The code worked part 1! \n\n This is the result {result }\n\n")
        generate_pdf_report(code, 'output.pdf')
        print("\n\n The code worked part 2 \n\n")
        return Command(
            update = {'error': "No Errors!"},
            goto= END
        )
    except Exception as e:
        print(f'\n\n This is the error {e} \n\n')
        return Command(
            update = {'error': e},
            goto= 'vis_agent'
        )