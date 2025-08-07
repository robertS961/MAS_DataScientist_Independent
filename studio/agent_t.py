
import re
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from agents import create_research_team, supervisor_team, machinelearning_agent, create_output_plotly_team
from classes import State
from helper_functions import pretty_print_messages, initialize_state_from_csv, define_variables, get_last_ai_message, get_datainfo, get_last_human_message, data_describe

load_dotenv()
class Agent:
    def __init__(self):
        self.ml_team = None
        self.research_team = None
        self.supervisor_team = None
        self.plotly_team = None

    def initialize(self):
        self.ml_team = machinelearning_agent(State)
        self.research_team = create_research_team(State)
        self.supervisor_team = supervisor_team(State)
        self.plotly_team = create_output_plotly_team(State)

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
        dic, config = define_variables(thread = 1, loop_limit = 10, data = data, data_info = data_info, name = "ml")
        
        #Get ML Team Ideas from the Data
        result = self.ml_team.invoke(dic, config)
        ideas_3 = get_last_ai_message(result['messages'])

        #Define the Class variables
        dic, config = define_variables(thread = 1, loop_limit = 10, data = data, data_info = data_info, name = "research")

        #Get Research Team Ideas from the Web
        result = self.research_team.compile(cache=MemorySaver()).invoke(input = dic, config = config)
        ideas_1 = get_last_ai_message(result['reflection']['messages'])
        
        #Get Supervisor Team Ideas from LLM's with different arrangement of MAS
        result = self.supervisor_team.compile(name = "supervisor_team", cache = MemorySaver()).invoke(input = dic, config = config)
        ideas_2 = get_last_ai_message(result['team_supervisor']['messages'])

        # Combine the above output to feed into a Reducer Agent which picks the best ideas then code + reflects til the code works!
        dic, config = define_variables(thread = 1, loop_limit = 25, data = data, data_info = data_info, name = "plotly", input = "\n".join([ideas_1, ideas_2,ideas_3]), data_description= data_description)
        dic['revise'] = True

        #Generate the Output for Reducing the ideas, Coding and Testing them, then improving the visualizations and testing them
        result = self.plotly_team.invoke(dic, config)
        msg = get_last_human_message(result['plotly_enhancer_leader']['messages'])
        
        #Return for total time
        return msg