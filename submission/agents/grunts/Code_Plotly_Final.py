import re
from langgraph.graph import END
from langgraph.types import Command
from langchain.schema import AIMessage
from classes import State
from typing import Literal
from helper_functions import run_code, get_last_ai_message
from langchain_experimental.utilities import PythonREPL

def code_plotly_final(state:State) -> Command[Literal['narrative_agent', 'vis_agent']]:
    last_ai = get_last_ai_message(state['messages'])
    state['code'] = last_ai
    code = last_ai
    try:
        print("🟢 Testing the code round 2...")
        run_code(code)
        print("✅ The Code Works!")
        return Command(
            update = {"messages":[
                {
                    "role": "user",
                    "content": (
                        f"Here is the code for the plotly graphs that runs without errors: \n {code} \n"
                        "Please enhance/add narratives to each graph through the HTML/Javascript that explain each graph, its importance, findings, and futures uses. Each narrative should be extremely educational and easy to understand! From each narrative highlight and bold the most important take away!\n"
                        "Make sure each paragraph flows into the new one like an academic paper \n"
                        "Please add a conclusion and intorduction at the start and end which are both a paragraph in length\n"
                        "Make sure each graph has a paragraph description about what the graph shows, why the algorithm was pick, what the algorithm does, how to interpret the graph and its results, and what would be impactful findings in the graph \n"
                        "Make sure all of these narratives get inserted into the HTML with the appropriate tags. The final output should include all of them! \n"
                        "Lastly make sure the entire HTML website flows well with the added narratives. IT should be presented like an educational report\n"
                        "Lastly take the ALL THE CODE  aka the python code and HTML/Javascript Code and print it to output.html file so it can be run on online servers! \n"
                        "Thank you \n"
                    )
                }
            ]},
            goto= 'narrative_agent'
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