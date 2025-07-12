from typing_extensions import TypedDict, Annotated
from langchain_core.messages import HumanMessage, SystemMessage, convert_to_messages
from langgraph.graph import StateGraph, START, END
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv
load_dotenv()




import csv
import operator
import numpy as np
from helpers import get_llm
from report_html import generate_html_report
from report_pdf import generate_pdf_report

class State(TypedDict):
    messages: Annotated[list, operator.add ]
    dataset_info: str
    iterations: int
    final_message: str
    report: str

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
    web_search = TavilySearch(max_results=3)
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
    visualization_agent = create_react_agent(
        model="openai:gpt-4o",
        tools=[],
        prompt=(
            "You are a visualiation and python coding agent.\n\n"
            "INSTRUCTIONS:\n"
            f"- Assist ONLY with visualization and code related tasks. You are to write clean code in Python to create beaitful data visualizations.\n"
            "- After you're done with your tasks, respond to the supervisor directly\n"
            "- Respond ONLY with the results of your work, do NOT include ANY other text."
        ),
        name="visualization_agent",
    )
    return visualization_agent

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


def create_workflow():
    supervisor = Supervisor_Agent().compile()
    for chunk in supervisor.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Imagine you are given a tabular dataset. Your goal to research novel data science and statistic ideas to perform on the data \n"
                    "Assign data exploration tasks to the both the stats_agent and the data science_agent \n"
                    "Then turn these ideas into python code that creates beautiful data visualizations \n"
                    "You should return code as the output \n",
                }
            ]
        },
    ):
        pretty_print_messages(chunk, last_message=True)

    final_message_history = chunk["supervisor"]["messages"]
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
        output_state = self.workflow.invoke(state)

        # flatten the output
        """
        def _flatten(value):
            return getattr(value, "content", value)
        result = {k: _flatten(v) for k, v in output_state.items()}
        """
        # Flatten the output since Langraph outputs AIMessage(content = " content here")
        def _flatten(value):
            return getattr(value, "content", value)

        result = {key: _flatten(field_value) for key, field_value in output_state.items()}
        
        print("----- Generated Code Output -----")
        print(result['report'])  
        print("---------------------------------")
        

        # decode the output
        self.decode_output(result)

        # return the result
        return result