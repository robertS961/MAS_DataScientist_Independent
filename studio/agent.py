
import re
import csv
import numpy as np

from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from agents import Swarm_Agent, create_judge, create_vis
from classes import State, Configurable
from helper_functions import pretty_print_messages, create_reflection_graph
from misc.report_pdf import generate_pdf_report

load_dotenv()

class Agent:
    def __init__(self):
        self.workflow = None

    def initialize(self):
        self.swarm = Swarm_Agent()
        print('swarm created \n\n')
        self.judge = create_judge()
        print('judge created \n\n')
        self.vis = create_vis()
        print('vis created \n\n')

    def initialize_state_from_csv(self) -> dict:
        # The dataset should be first input to the agentic configuration, and it should be generalizable to any dataset
        path = "./dataset.csv"
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)
            first_row = next(reader)

        attributes = ", ".join(header)
        example_values = "\t".join(first_row)

        # if the final output contains Vega-Lite codes, then use the github hosted dataset rather than the local dataset
        # file_name = "https://raw.githubusercontent.com/demoPlz/mini-template/main/studio/dataset.csv"
        file_name = "dataset.csv"
        example_input = f"""
            There is a dataset, there are the following {len(header)} attributes:
            {attributes}
            Name of csv file is {file_name}
        """
        state = {
            "dataset_info": str(example_input)
        }

        return state
    
    def decode_output(self, output: dict):
        # if the final output contains Vega-Lite codes, then use generate_html_report
        # if the final output contains Python codes, then use generate_pdf_report

        generate_pdf_report(output, "output.pdf")
        # generate_html_report(output, "output.html")
    
    def define_variables(self, thread: int, loop_limit: int, data:str):
        #Define the Prompt to insert into the LLM
        prompt= (f"You are given a tabular dataset. Here is the data {data} \n"
                "Your goal to research novel data science and statistic ideas to perform on the data \n"
                "Swarm all the agents until novel data science and statistic learning ideas are created based on the tabular dataset given above!\n"
                "The ideas should be clearly labeled and readable \n\n"
                "This data will be passed on for evulation so please make it the best you can! Thank you \n\n"
        )
        #Orginal State Variables for Invoke
        dic: State = {"messages": [
            {
                "role": "user",
                "content": f"{prompt}"
            }
            ],
            "dataset_info": data,
            "error": "",
            "ideas": "",
            'flag': True,
        }
        #Orginal Configurations for our graph
        config :Configurable = {"thread_id": thread, "recursion_limit": loop_limit}
        return dic, config

    def process(self):

        #Return error if any of the graphs didn't build
        if self.swarm is None or self.vis is None or self.judge is None:
            raise RuntimeError("Agent not initialised. Call initialize() first.")
        
        # initialize the state & read the dataset
        state = self.initialize_state_from_csv()

        # invoke the workflow
        data = state['dataset_info']

        #Define the Class variables
        dic, config = self.define_variables(thread = 1, loop_limit = 6, data = data)
       
        # Create a reflection step from the swarm to repeating itself
        reflection_app = create_reflection_graph(self.swarm, self.judge, self.vis, State)
        
        #Stream the Output
        for chunk in reflection_app.compile(cache=MemorySaver()).stream(input = dic, config = config):
            pretty_print_messages(chunk, last_message=True)
        
        #Find the last message
        result = chunk['visualization']['messages'][-1].content
       
        #Put the output in a python file 
        code = re.findall(r"```python\n(.*?)\n```", result, re.DOTALL)
        with open("extracted_code.py", "a", encoding="utf-8") as f:
            f.write("# ---- NEW BLOCK ---- # \n")
            for block in code:
                f.write(block + "\n\n")
        
        # decode the output
        self.decode_output(result)

        # return the result
        return result