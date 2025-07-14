from typing_extensions import TypedDict, Annotated
from langchain_core.messages import HumanMessage, SystemMessage, convert_to_messages
from langgraph.graph import StateGraph, START, END
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv
load_dotenv()



import re
import csv
import operator
import numpy as np
import json
import pickle
from helpers import get_llm
from report_html import generate_html_report
from report_pdf import generate_pdf_report

class State(TypedDict):
    messages: str
    dataset_info: str
    #iterations: int
    #final_message: str
    #report: str

class FinalState(TypedDict):
    final_report: str
    dataset_info: str

def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")

def WebSearch():
    web_search = TavilySearch(max_results=1)
    return web_search
    

def Research_DataScience_Agent():
    web_search = WebSearch()
    research_ds_agent = create_react_agent(
        model="openai:gpt-4o",
        tools=[web_search],
        prompt=(
            "You are a research data science agent.\n\n"
            "INSTRUCTIONS:\n"
            "- Assist ONLY with research-related tasks, your task is to discover novel data science strategies. \n"
            "- After you're done with your tasks, respond to the supervisor directly\n"
            "- Respond ONLY with the results of your work, do NOT include ANY other text."
        ),
        name="research_ds_agent",
    )
    return research_ds_agent

def Research_Stat_Agent():
    web_search = WebSearch()
    research_stat_agent = create_react_agent(
        model="openai:gpt-4o",
        tools=[web_search],
        prompt=(
            "You are a research statistician agent.\n\n"
            "INSTRUCTIONS:\n"
            "- Assist ONLY with research-related tasks, your task is to discover novel statistic ideas to perform on tabular datasets. \n"
            "- After you're done with your tasks, respond to the supervisor directly\n"
            "- Respond ONLY with the results of your work, do NOT include ANY other text."
        ),
        name="research_stat_agent",
    )
    return research_stat_agent

def Visualization_Agent():
    tools = []
    prompt_template = (
        "You are a Python visualization expert. You generate stunning visualizations using matplotlib, seaborn, or plotly.\n\n"
        "DATASET INFO:\n{dataset_info}\n\n"
        "Respond ONLY with Python code that creates meaningful and aesthetic data visualizations. Do not explain anything. Thank you \n\n"
        "Make sure the Python code runs and there isn't any bugs. Also write it in simple python code so the users can easily understand it \n\n"
    )
    def custom_prompt(state: dict) -> str:
        return prompt_template.format(
            dataset_info=state.get("dataset_info", "")
        )
    return create_react_agent(
        model="openai:gpt-4o",
        tools=tools,
        prompt= prompt_template,
        name="visualization_agent",
    )
    

def Supervisor_Agent():
    research_ds_agent = Research_DataScience_Agent()
    research_stat_agent = Research_Stat_Agent()
    visualization_agent = Visualization_Agent()
    supervisor = create_supervisor(
    model=init_chat_model("openai:gpt-4o"),
    agents=[research_ds_agent, research_stat_agent, visualization_agent],
    prompt=(
        "You are a supervisor managing three agents:\n"
        "- a research data science agent. Assign research-related data science tasks to this agent\n"
        "- a research statistician agent. Assign research-related statistic tasks to this agent"
        "- a visualization and coding agent. Assign coding tasks for visualization to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
    add_handoff_back_messages=True,
    output_mode="full_history", )
    return supervisor

def translate_node(state:FinalState):
    final_report = state['final_report']
    dataset_info = state['dataset_info']
    prompt = (
        "You are a senior level data analyst. You excell at python programming especially in data visualization \n"
        f"Here is the current state of the analysis:\n{final_report}\n\n"
        f"The dataset description is:\n{dataset_info}\n\n"
        "There is a lot of python code in the output along with some useless text from the previous agents. Parse for all the python code first \n"
        "Then transform the python text into readbale and runable python code. Then improve it to make the data visualizations better \n"
        "Please improve and extend the Python code and narrative to make the analysis deeper and more impactful. Output should be Python code with graphs and written insights."
    )
    human_prompt = (
        "Please make sure the python code runs without bugs. Then double check to make sure that it runs and that the images are easily visable and don't contain emepty graphs or unreadable figures \n\n"
        "Also make sure the code doesn't contain any non coding lines. IT will be put through a coding translator which can only take Python code! Thank you \n\n"
        "Lastly please make the code simple and easy to read!"
    )

    llm = get_llm(temperature=0, max_tokens=4096)
    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content= human_prompt)
    ])
    return {"final_report": response}



def create_finalReport():
    builder = StateGraph(FinalState)
    builder.add_node('translate_node', translate_node)
    
    builder.add_edge(START, 'translate_node')
    builder.add_edge('translate_node', END)

    return builder.compile()


def create_workflow():
    return Supervisor_Agent().compile()
    

    '''
    create the agentic workflow using LangGraph
    builder = StateGraph(State)
    builder = generate_team(builder)
    return builder.compile()
    '''

    

class Agent:
    def __init__(self):
        self.workflow = None

    def initialize(self):
        self.workflow = create_workflow()

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
    def process(self):

        if self.workflow is None:
            raise RuntimeError("Agent not initialised. Call initialize() first.")
        
        # initialize the state & read the dataset
        state = self.initialize_state_from_csv()
        # invoke the workflow
        #output_state = self.workflow.invoke(state)
        prompt= (f"You are given a tabular dataset. Here is the data {state["dataset_info"]} \n"
                "Your goal to research novel data science and statistic ideas to perform on the data \n"
                "Assign data exploration tasks to the both the stats_agent and the data science_agent \n"
                f"Then turn these ideas into python code that creates beautiful data visualizations using the {state['dataset_info']} \n"
                "You should return code as the output \n")

        dic = {"messages": [
            {
                "role": "user",
                "content": f"{prompt}"
            }
            ]
        }
        
        for chunk in self.workflow.stream(input = dic): # Label this state better
            pretty_print_messages(chunk, last_message=True)

        # Hold the final message from supervisor
        final_message_history = chunk
        
        # Let's say your huge dict is called `result_dict`
        result_text = final_message_history["supervisor"]["messages"][-1].content
        
        # Extract everything that looks like code inside triple backticks
        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", result_text, re.DOTALL)

        # Write each block to a separate file or combine into one
        with open("extracted_code.py", "w", encoding="utf-8") as f:
            for block in code_blocks:
                f.write(block + "\n\n")

        # Write the entire block to a file for debugging purposes
        with open("debug_output.txt", "w", encoding="utf-8") as f:
            f.write(final_message_history['supervisor']['messages'][-1].content)
        '''
        final_state = {
            'final_report': final_message_history,
            'dataset_info': state['dataset_info']
        }
        final = create_finalReport()
        output_state = final.invoke(final_state)
        print(output_state)
        '''
        
        '''with open("debug_output.txt", "w", encoding="utf-8") as f:
            f.write(values)'''
        #create_output(supervisor)

   
        # Flatten the output since Langraph outputs AIMessage(content = " content here")
        '''
        def _flatten(value):
            return getattr(value, "content", value)

        result = {key: _flatten(field_value) for key, field_value in output_state.items()}
        '''
        '''
        print("----- Generated Code Output -----")
        print(result['final_report'])  
        print("---------------------------------")
        '''
        
        # decode the output
        self.decode_output(result_text)

        # return the result
        return code_blocks