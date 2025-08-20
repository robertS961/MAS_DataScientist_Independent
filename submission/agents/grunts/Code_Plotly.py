import re
from langgraph.graph import END
from langgraph.types import Command
from langchain.schema import AIMessage
from classes import State
from typing import Literal
from helper_functions import run_code, get_last_ai_message
from langchain_experimental.utilities import PythonREPL

def code_plotly(state:State) -> Command[Literal[END, 'vis_agent']]:
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
                    "role": "user",
                    "content": (
                        f"Here is the code for the plotly graphs that runs without errors: \n {code} \n"
                        "Your goal is to improve this code by making the plotly graphs more interactive and beautiful! Currently some of them look blank and don't show any meaningful data! \n"
                        "Make sure everything is easy to see. NO overlapping buttons with button description, NO overlapping buttons with graph keys, NO BUTTON DESCRIPTIONS that disppear on selection, and NO white out hover/selected buttons!\n"
                        "Make sure the code runs without errors and bugs! \n"
                        "Return the full python code with plotly and the html/javascript code at the end! \n"
                        "Thank you ! \n"
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