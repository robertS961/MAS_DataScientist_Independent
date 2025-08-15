
import re
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from agents import create_research_team, supervisor_team, machinelearning_agent, create_output_plotly_team
from classes import State
from helper_functions import pretty_print_messages, initialize_state_from_csv, define_variables, get_last_ai_message, get_datainfo, data_describe, run_code

load_dotenv()
class Agent:
    def __init__(self):
        self.workflow = None

    def initialize(self):
        self.ml_team = machinelearning_agent(State)
        print("Machine Learning Team Created \n")
        self.research_team = create_research_team(State)
        print("Research Team Created \n")
        self.supervisor_team = supervisor_team(State)
        print("Supervisor Team is Created! \n")
        self.plotly_team = create_output_plotly_team(State)
        print("Plotly Team is Created! \n")
    '''    
    def decode_output(self, output: dict):
        # if the final output contains Vega-Lite codes, then use generate_html_report
        # if the final output contains Python codes, then use generate_pdf_report
        generate_pdf_report(output, "output.pdf")
        # generate_html_report(output, "output.html")
    '''
    def process(self):
        #Return error if any of the graphs didn't build
        if self.research_team is None or self.supervisor_team is None or self.plotly_team is None or self.ml_team is None:
            raise RuntimeError("Agent not initialised. Call initialize() first.")
        
        # initialize the state & read the dataset
        path = "dataset.csv" # change this when you change the dataset
        state = initialize_state_from_csv()
        data_info = get_datainfo(path)
        data = state['dataset_info']
        data_description = data_describe(path)

        #Set up State Variables for the ML Team
        dic, config = define_variables(thread = 1, loop_limit = 25, data = data, data_info = data_info, name = "ml")
        
        #Get ML Team Ideas from the Data
        result = self.ml_team.invoke(dic, config)
        ideas_3 = get_last_ai_message(result['messages'])

        #Define the Class variables
        dic, config = define_variables(thread = 1, loop_limit = 25, data = data, data_info = data_info, name = "research")
    
        #Stream the Output for Generating Research Ideas from the Web
        for chunk in self.research_team.compile(cache=MemorySaver()).stream(input = dic, config = config):
            pretty_print_messages(chunk, last_message=True)
        
        ideas_1 = get_last_ai_message(chunk['reflection']['messages'])
        
        #Stream the Output for Generating Research Ideas from LLM's with different arrangement of MAS
        for chunk in self.supervisor_team.compile(name = "supervisor_team", cache = MemorySaver()).stream(input = dic, config = config):
            pretty_print_messages(chunk, last_message=True)

        ideas_2 = get_last_ai_message(chunk['team_supervisor']['messages'])

        # Combine the above output to feed into a Reducer Agent which picks the best ideas then code + reflects til the code works!
        dic, config = define_variables(thread = 1, loop_limit = 25, data = data, data_info = data_info, name = "plotly", input = "\n".join([ideas_1, ideas_2,ideas_3]), data_description= data_description)
        dic['revise'] = True
        
        #Generate the Output for Reducing the ideas, Coding and Testing them, then improving the visualizations and testing them
        for chunk in self.plotly_team.stream(dic, config):
            pretty_print_messages(chunk)

        #Get the final code 
        msg = get_last_ai_message(chunk['plotly_enhancer_leader']['messages'])

        #Add Code to Data Base for testing
        code = re.findall(r"```python\n(.*?)\n```", msg, re.DOTALL)

        #Write the code to a file for testing
        with open("test_plotly_code.py", "w", encoding="utf-8") as f:
            f.write("# ---- NEW BLOCK ---- # \n")
            for block in code:
                f.write(block + "\n\n")

        #Generate PDF output
        #generate_pdf_report(msg, 'output.pdf')
        #Return to Run.py
        run_code(msg)
        return msg