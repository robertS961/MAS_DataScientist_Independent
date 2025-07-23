
import re
import csv
import numpy as np

from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from agents import Swarm_Agent, create_judge, create_vis, create_research_team, supervisor_team
from classes import State, Configurable, TempState
from helper_functions import pretty_print_messages, create_reflection_graph, initialize_state_from_csv, define_variables, get_last_ai_message
from misc.report_pdf import generate_pdf_report



load_dotenv()

class Agent:
    def __init__(self):
        self.workflow = None

    def initialize(self):
        self.research_team = create_research_team(State)
        print("Research Team Created \n")
        self.supervisor_team = supervisor_team(State)
        print("Supervisor Team is Created! \n")
        self.vis = create_vis()
        print('vis created \n\n')

    
    def decode_output(self, output: dict):
        # if the final output contains Vega-Lite codes, then use generate_html_report
        # if the final output contains Python codes, then use generate_pdf_report
        generate_pdf_report(output, "output.pdf")
        # generate_html_report(output, "output.html")

    def process(self):

        #Return error if any of the graphs didn't build
        if self.research_team is None or self.supervisor_team is None:
            raise RuntimeError("Agent not initialised. Call initialize() first.")
        
        # initialize the state & read the dataset
        state = initialize_state_from_csv()

        # invoke the workflow
        data = state['dataset_info']

        #Define the Class variables
        dic, config = define_variables(thread = 1, loop_limit = 10, data = data)
    
        #Stream the Output
        for chunk in self.research_team.compile(cache=MemorySaver()).stream(input = dic, config = config):
            pretty_print_messages(chunk, last_message=True)
        
        ideas_1 = get_last_ai_message(chunk['reflection']['messages'])
        #print(f"\n\n {last_ai_content}\n\n ")

        for chunk in self.supervisor_team.compile(name = "supervisor_team", cache = MemorySaver()).stream(input = dic, config = config):
            pretty_print_messages(chunk, last_message=True)

        ideas_2 = get_last_ai_message(chunk['team_supervisor']['messages'])
        #print(f"\n\n This is the last_ai_content round 2 {last_ai_content_2} \n\n")
        
        #Find the last message
        #print(f"\n\n {chunk} \n\n")

        #result = chunk['visualization']['messages'][-1].content
       
        #Put the output in a python file
        '''
        code = re.findall(r"```python\n(.*?)\n```", result, re.DOTALL)
        with open("extracted_code.py", "a", encoding="utf-8") as f:
            f.write("# ---- NEW BLOCK ---- # \n")
            for block in code:
                f.write(block + "\n\n")
        '''
        with open('written.txt', 'a', encoding= "utf-8" ) as f:
            f.write(ideas_1 + '\n \n')
            f.write(ideas_2 + '\n \n')
        # decode the output
        #self.decode_output(result)

        # return the result
        return [ideas_1, ideas_2]