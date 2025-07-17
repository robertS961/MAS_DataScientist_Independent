from typing_extensions import TypedDict, Annotated
from langchain_core.messages import HumanMessage, SystemMessage, convert_to_messages
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_tavily import TavilySearch
from langgraph.managed import RemainingSteps
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.cache.memory import InMemoryCache
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langgraph_swarm import create_swarm, create_handoff_tool
from langgraph_reflection import create_reflection_graph
from openevals.llm import create_llm_as_judge
from typing import List, Optional, Type, Any, get_type_hints, Literal
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
    messages: List[str] # Might need to change to str
    feedback: List[str]
    dataset_info: str
    #iterations: int
    #final_message: str
    #report: str

class FinalState(TypedDict):
    code: str
    error: str
    flag: bool
    dataset_info: str


class Finish(TypedDict):
    """Tool for the judge to indicate the response is acceptable."""

    finish: bool



class MessagesWithSteps(State):
    remaining_steps: RemainingSteps

def end_or_reflect(state: MessagesWithSteps) -> Literal["visualization", "graph"]:
    print(state["remaining_steps"], len(state["messages"]))
    if state["remaining_steps"] <= 2:
        return "visualization"
    if len(state["messages"]) <= 0:
        return "visualization"
    return "graph"
    


def create_reflection_graph(
    graph: CompiledStateGraph,
    reflection: CompiledStateGraph,
    visualization: CompiledStateGraph,
    state_schema: Optional[Type[Any]] = None,
    config_schema: Optional[Type[Any]] = None,
) -> StateGraph:
    _state_schema = state_schema or graph.builder.schema
    
    if "remaining_steps" in _state_schema.__annotations__:
        raise ValueError(
            "Has key 'remaining_steps' in state_schema, this shadows a built in key"
        )
    
    if "messages" not in _state_schema.__annotations__:
        raise ValueError("Missing required key 'messages' in state_schema")

    class StateSchema(_state_schema):
        remaining_steps: RemainingSteps

    rgraph = StateGraph(StateSchema, config_schema=config_schema)
    rgraph.add_node("graph", graph)
    rgraph.add_node("reflection", reflection)
    rgraph.add_node('visualization', visualization)
    rgraph.add_edge(START, "graph")
    rgraph.add_edge("graph", "reflection")
    rgraph.add_conditional_edges("reflection", end_or_reflect)
    return rgraph

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

def decode_output(output):
        # if the final output contains Vega-Lite codes, then use generate_html_report
        # if the final output contains Python codes, then use generate_pdf_report

        generate_pdf_report(output, "output.pdf")
        # generate_html_report(output, "output.html")

def WebSearch():
    """ This function searches the web for relevant research information! """
    web_search = TavilySearch(max_results=1)
    return web_search

def transfer_to_ds_agent():
    transfer_to_ds_agent = create_handoff_tool(
        agent_name = 'ds_assistant',
        description = (
            'Transfer user to a data science agent! Capabale of coming up with novel data science ideas \n\n'
            'It uses a websearch tool Tavily to help find research ideas for data science \n\n'
        ),
    )
    return transfer_to_ds_agent

def transfer_to_stats_agent():
    transfer_to_stats_agent = create_handoff_tool(
        agent_name = 'stats_assistant',
        description = (
            'Transfer user to a statistician agent! Capabale of coming up with novel statistical learning ideas \n\n'
            'It uses a websearch tool Tavily to help find research ideas for statisticial learning and statistics \n\n'
        ),
    )
    return transfer_to_stats_agent

def transfer_to_vis_agent():
    transfer_to_vis_agent = create_handoff_tool(
        agent_name = 'vis_assistant',
        description = (
            'Transfer user to a visualization agent! Capabale of taking the previous ideas and turning them into computable python code! \n\n'
            'It uses the LLM enginge to take those ideas and create clean, bug free, and simple python code! \n\n'
        ),
    )
    return transfer_to_vis_agent


    

def Research_DataScience_Agent(state:State):
    web_search = WebSearch()
    stats_agent = transfer_to_stats_agent()
    previous_message = state['messages']
    previous_feedback = state['feedback']
    data = state['dataset_info']
    research_ds_agent = create_react_agent(
        model="openai:gpt-4o",
        tools=[web_search, stats_agent],
        prompt=(
            "You are a research data science agent.\n\n"
            "INSTRUCTIONS:\n"
            "Your goal is find novel data science ideas for a tabular dataset. These ideas should be able to be applied to tabular data .\n\n"
            f"You might have a previous messaged from another agent it is: {previous_message}"
            "You might also have feedback message from another agent to improve upon the previous_message and your soon to be future message. \n\n"
            f"If there is feedback it is: {previous_feedback}\n\n"
            f"These are the columns of the tabular data. Use them in the ideas by citing the column names: {data} \n\n"
            "You have access to two different tools! \n\n"
            "You can web_search the internet to discover ideas! \n\n"
            "You can also transfer you knowledge to the statistican agent \n\n"
            "Once you discover interesting and novel ideas then list them! Make sure you use the feedback in the ideas if there is any! \n\n"
            "Come up with at least 5 ideas before you transfer the user to the stats agent. Thank you!!! \n\n"
        ),
        name="ds_assistant",
    )
    return research_ds_agent

def Research_Stat_Agent(state:State):
    web_search = WebSearch()
    ds_agent = transfer_to_ds_agent()
    previous_message = state['messages']
    previous_feedback = state['feedback']
    data = state['dataset_info']
    #vis_agent = transfer_to_vis_agent()
    research_stat_agent = create_react_agent(
        model="openai:gpt-4o",
        tools=[web_search, ds_agent],
        prompt=(
            "You are a research statistician agent.\n\n"
            "INSTRUCTIONS:\n"
            "Your goal is find novel statistical learning ideas for a tabular dataset."
            f"You might have a previous messaged from another agent it is: {previous_message}"
            "You might also have feedback message from another agent to improve upon the previous message and your soon to be future message. \n\n"
            f"If there is feedback it is: {previous_feedback} \n\n"
            f"These are the columns of the tabular data. Use them in the ideas by citing the column names: {data} \n\n"
            "You have access to two different tools! \n\n"
            "You can web_search the internet to discover ideas! \n\n"
            "You can also call on the ds (Data science agent) \n\n"
            "Feel free to discover interesting ideas on your own, add to the ideas of the previous message, and search the web! \n\n!"
            "If there is feedback make sure you use this to improve your results espeically adding onto other ideas and creating new ideas! \n\n"
            "Return all the ideas in a list. Make sure there is at least 6 ideas then transfer the user to the ds(data science) agent\n\n"
            "Thank you! \n\n"
        ),
        name="stats_assistant",
    )
    return research_stat_agent


def vis_a(state:State):
    tools = []
    data = state['dataset_info'] if state['dataset_info'] else ""
    ideas = state['messages'][-1] if state['messages'] else ""
    prompt = (
        "You are a Python visualization expert. You generate stunning visualizations using matplotlib, seaborn, plotly, or any libraries you can think of!.\n\n"
        "We have already created several ideas. Your objective is to take the ideas and code them ! \n\n"
        f"The dataset columns are as follows: {data}\n\n"
        f"The ideas are as follows: {ideas}\n\n"
        "Respond ONLY with Python code that creates meaningful and aesthetic data visualizations. Do not explain anything. Thank you \n\n"
        "Make sure the Python code runs and there isn't any bugs. Also write it in simple python code so the users can easily understand it \n\n"
    )
    model = ChatOpenAI(model="gpt-4o", temperature=0,  max_tokens=4096)
    response = model.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"Generate the code from these ideas {ideas}. Please make sure it runs and it is in python without bugs! Double check that it runs without error. In the past the code would not run because of errors! Thank you \n\n")
    ])
    print(response)
    return {'messages':[{"role": "user", "content": str(response.content)}]}


def Swarm_Agent(state:State):
    
    research_ds_agent = Research_DataScience_Agent(State)
    print('created research_ds_agent\n\n')
    research_stat_agent = Research_Stat_Agent(State)
    #reserarch_vis_agent = Visualization_Agent()
    swarm_agent = create_swarm(
    agents=[research_ds_agent, research_stat_agent],
    default_active_agent = 'ds_assistant',
    )
    return swarm_agent #create the swarm everytime and input the new state from the other graph


def make_judge(state:State):
    current_message = state['messages']
    critique_prompt = (
        
        "-You are a senior data scientist. You have over 20 years of experience! \n\n"
        "- You are going to be given data science and statistical learning ideas that can be applied to tabular data! \n\n"
        "Use your experience to state if it uses modern day data science practices \n\n"
        "If there is any room for improvement please suggest it back to the team of data scientist! \n\n"
        "If the response meets the ALL the above criteria, return an empty string"
        "If you find ANY issues with the response, do not return an empty string. Instead, provide specific and constructive feedback which should return as a string."
        "Be detailed in your critique so the team can understand exactly how to improve."

    )

    """Evaluate the assistant's response using a separate judge model."""
    model = ChatOpenAI(model = 'gpt-4o', temperature = 0, max_tokens = 4096)
    response = model.invoke([
        SystemMessage(content=critique_prompt),
        HumanMessage(content=f"Here is the current message{current_message} \n\n. State wether this code meets the protocol. If false return a message so a string with all the improvements. If True return an empty string \n\n")
    ])
    print(f" \n\n These are the eval_result : {response} \n\n")

    if response == "":
        print("✅ Response approved by judge")
        return
    else:
        # Otherwise, return the judge's critique as a new user message
        print("⚠️ Judge requested improvements")
        return {"feedback": [{"role": "user", "content": str(response.content)}]}

def validate_code(state:FinalState):
    code = state['code']
    #print(f"\n\nThis is the content of the code {code[0]['content']}\n\n")
    print(f"Normal output code{code}\n\n")
    try:
        decode_output(code)
        print('WInnner \n\n')
        return {'flag':True}
    except Exception as e:
        print(f"{e}\n\n")
        return {'error':[{"role": "user", "content": f"⚠️ The code failed with error:\n```\n{str(e)}\n```\n Please fix it and try again."}]}


def fix_code(state:FinalState):
    code = state['code']
    error = state['error']
    columns = state['dataset_info']
    prompt = (
        "You are an expert Python coder! You are senior level with ove 15 years of experience with coding in Python!.\n\n"
        f"Code has already been generated for you and it is here{code} \n\n"
        f"The following error occured {error} and these are the dataset columns {columns}\n\n"
        "Please fix the code and make sure it runs without error\n\n"
        "Respond ONLY with Python code. Thank you \n\n"
        "Make sure the Python code runs and there isn't any bugs. Also write it in simple python code so the users can easily understand it \n\n"
    )
    model = ChatOpenAI(model="gpt-4o", temperature=0,  max_tokens=4096)
    response = model.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"Please fix this {code}! Return the fixed code without any bugs or new bugs. Please run it to make sure it works! Thank you \n\n")
    ])
    print(f'\n\nThis is the model response {response} \n\n')
    return {'code':[{"role": "user", "content": str(response.content)}]}





def create_swerm():
    return Swarm_Agent(State).compile()

def create_vis():
    return(
        StateGraph(State)
        .add_node('vis_agent', vis_a)
        .add_edge(START, 'vis_agent')
        .add_edge('vis_agent', END)
        .compile()
    )

def create_judge():
    return(
        StateGraph(State)
        .add_node('judge_agent', make_judge)
        .add_edge(START, 'judge_agent')
        .add_edge('judge_agent', END)
        .compile()
    )

def create_coder():
    return (
        StateGraph(FinalState)
        .add_node('validate_node', validate_code)
        .add_node('fix_node', fix_code)
        .add_edge(START, 'validate_node')
        .add_conditional_edges('validate_node', lambda state: state['flag'] == True, {
            True: END,
            False: 'fix_node'
        })
        .add_edge('fix_node', 'validate_node')
        .compile()
    )




    

class Agent:
    def __init__(self):
        self.workflow = None

    def initialize(self):
        self.swarm = create_swerm()
        print('swarm created \n\n')
        self.judge = create_judge()
        print('judge created \n\n')
        self.vis = create_vis()
        print('vis created \n\n')
        self.code = create_coder()
        print('coder created \n\n')

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
    def process(self):

        if self.swarm is None or self.vis is None or self.judge is None or self.code is None:
            raise RuntimeError("Agent not initialised. Call initialize() first.")
        
        # initialize the state & read the dataset
        state = self.initialize_state_from_csv()
        # invoke the workflow
        #output_state = self.workflow.invoke(state)
        prompt= (f"You are given a tabular dataset. Here is the data {state["dataset_info"]} \n"
                "Your goal to research novel data science and statistic ideas to perform on the data \n"
                "Swarm all the agents until novel data science and statistic learning ideas are created based on the tabular dataset given above!\n"
                "The ideas should be clearly labeled and readable \n\n"
                "This data will be passed on for evulation so please make it the best you can! Thank you \n\n"
        )

        dic = {"messages": [
            {
                "role": "user",
                "content": f"{prompt}"
            }
            ],
            "dataset_info": state['dataset_info']
        }

        reflection_app = create_reflection_graph(self.swarm, self.judge, self.vis, State)
        for chunk in reflection_app.compile(cache=InMemoryCache()).stream(dic, {"recursion_limit": 6}):
            pretty_print_messages(chunk, last_message=True)
        
        result = chunk['visualization']['messages'][0]['content']

        code = re.findall(r"```python\n(.*?)\n```", result, re.DOTALL)
        with open("extracted_code.py", "a", encoding="utf-8") as f:
            f.write('\n\n#--- New Code Block --- \n')
            for block in code:
                f.write(block + "\n\n")
        print(f"\n\n This is the result code {result}\n\n")
        final_state = {'code' : result, 'error': "", 'dataset_info': state['dataset_info'],' flag': False}
        res = self.code.invoke(final_state)
        return res

        '''
        for chunk in self.workflow.stream(input = dic): # Label this state better
            pretty_print_messages(chunk, last_message=True)
        # Hold the final message from supervisor
        '''
        #final_message_history = chunk['vis_assistant']['messages'][-1].content
        '''
        with open("debug_output.txt", "w", encoding="utf-8") as f:
            f.write(str(result))
        '''

        #Put the output in a python file 
        
        
        # Let's say your huge dict is called `result_dict`
        # result_text = final_message_history["stats_assistant"]["messages"]['HumanMessage'].content
        '''
        # Extract everything that looks like code inside triple backticks
        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", result_text, re.DOTALL)
        '''
        # Write each block to a separate file or combine into one
        '''
        with open("extracted_code.py", "w", encoding="utf-8") as f:
            for block in matches:
                f.write(block + "\n\n")
        '''
        '''
        # Write the entire block to a file for debugging purposes
        with open("debug_output.txt", "w", encoding="utf-8") as f:
            f.write(result_text)
        '''
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
        

        

        # return the result