from langgraph.graph import END
from langgraph.types import Command
from classes import State
from typing import Literal
from helper_functions import generate_pdf_report

def code_agent(state:State) -> Command[Literal[END, 'vis_agent']]:
    code = state['messages'][-1].content
    print(f'\n\n this is the code {code} \n\n')
    try:
        generate_pdf_report(code, 'output.pdf')
        print('\n\n The code works!! \n\n')
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