import re
from langgraph.graph import END
from langgraph.types import Command
from langchain.schema import AIMessage
from classes import State
from typing import Literal
from helper_functions import run_code, get_last_ai_message
from langchain_experimental.utilities import PythonREPL

def code_narrative(state:State) -> Command[Literal['vis_narrative', END]]:
    last_ai = get_last_ai_message(state['messages'])
    state['code'] = last_ai
    code = last_ai
    try:
        print("🟢 Testing the code ...")
        run_code(code)
        print("✅ The Code Works!")
        return Command(
            update = {"messages":[
                {
                    "role": "ai",
                    "content": f"{code}"
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
            goto= 'vis_narrative'
        )