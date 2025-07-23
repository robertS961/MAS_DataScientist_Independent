
import re
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from agents import create_research_team, supervisor_team, create_code
from classes import State
from helper_functions import pretty_print_messages, initialize_state_from_csv, define_variables, get_last_ai_message
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
        self.code_team = create_code(State)
        print(" Coding Team is Created! \n")
        

    
    def decode_output(self, output: dict):
        # if the final output contains Vega-Lite codes, then use generate_html_report
        # if the final output contains Python codes, then use generate_pdf_report
        generate_pdf_report(output, "output.pdf")
        # generate_html_report(output, "output.html")

    def process(self):

        #Return error if any of the graphs didn't build
        if self.research_team is None or self.supervisor_team is None or self.code_team is None:
            raise RuntimeError("Agent not initialised. Call initialize() first.")
        
        # initialize the state & read the dataset
        state = initialize_state_from_csv()

        # invoke the workflow
        data = state['dataset_info']

        #Define the Class variables
        dic, config = define_variables(thread = 1, loop_limit = 10, data = data, name = "research")
    
        #Stream the Output for Generating Research Ideas from the Web
        for chunk in self.research_team.compile(cache=MemorySaver()).stream(input = dic, config = config):
            pretty_print_messages(chunk, last_message=True)
        
        ideas_1 = get_last_ai_message(chunk['reflection']['messages'])
        
        #Stream the Output for Generating Research Ideas from LLM's with different arrangement of MAS
        for chunk in self.supervisor_team.compile(name = "supervisor_team", cache = MemorySaver()).stream(input = dic, config = config):
            pretty_print_messages(chunk, last_message=True)

        ideas_2 = get_last_ai_message(chunk['team_supervisor']['messages'])

        # Combine the above output to feed into a Reducer Agent which picks the best ideas then code + reflects til the code works!
        dic, config = define_variables(thread = 1, loop_limit = 25, data = data, name = "code", input = "\n".join([ideas_1, ideas_2]))

        #Generate the Output for Reducing the ideas and then generating code plus solving errors
        for chunk in self.code_team.stream(input = dic, config = config):
            pretty_print_messages(chunk)

        result = chunk['code_agent']['messages'][-1]['content']

        #Put the code into a seperate file for research purposes later!
        code = re.findall(r"```python\n(.*?)\n```", result, re.DOTALL)
        print(f"This is the code ! \n {code} \n")
        with open("extracted_code.py", "a", encoding="utf-8") as f:
            f.write("# ---- NEW BLOCK ---- # \n")
            for block in code:
                f.write(block + "\n\n")

        #Generate Local Output in pdf format
        generate_pdf_report(result, 'output.pdf')

        #Return to Run.py
        return result
    
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
        '''
        with open('written.txt', 'a', encoding= "utf-8" ) as f:
            f.write(ideas_1 + '\n \n')
            f.write(ideas_2 + '\n \n')
        '''
        # decode the output
        #self.decode_output(result)