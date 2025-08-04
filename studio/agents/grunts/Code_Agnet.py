import re
from langgraph.graph import END
from langgraph.types import Command
from langchain.schema import AIMessage
from classes import State
from typing import Literal
from helper_functions import generate_pdf_report, get_last_ai_message
from langchain_experimental.utilities import PythonREPL

def code_agent(state:State) -> Command[Literal[END, 'vis_agent']]:
    #repl = PythonREPL()
    code = get_last_ai_message(state['messages'])
    state['code'] = code
    #print(f"\n This is the code \n {code} \n")
    try:
        # Regex to extract content inside triple single quotes
        #py_code = re.findall(r"```python\n(.*?)```", code, re.DOTALL)
        #print(f'\n\n this is the code {py_code[0]} \n\n')
        #result = repl.run(py_code[0])
        #print(f"\n\n The code worked part 1! \n\n This is the result {result }\n\n")
        generate_pdf_report(code, 'output.pdf')
        print("✅ The Code Works!")
        return Command(
            update = {"messages":[
                {
                    "role": "user",
                    "content": (
                        f"You are given code for a pdf.  Here it is \n {code} \n"
                        "Your goal is to evulate this pdf and make sure each visualization is clear to understand!\n"
                        "Focus on clear titles, axis, correct sizes figures/designs, colors, and keys for uncertain figures \n"
                        "Thank you !"
                    )
                }
            ]},
            goto= END
        )
    except Exception as e:
        print(f"⚠️ The Code has Errors. They are \n {e} \n ")
        message = f"This is the error for the code given, please fix the code to this error! \n {e} \n"
        state['errors'] = e
        return Command(
            update = {"messages":[
                {
                    "role": "ai",
                    "content": f"{message}"
                }
            ]},
            goto= 'vis_agent'
        )