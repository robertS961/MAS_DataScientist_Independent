from classes import State
from langgraph.prebuilt import create_react_agent
from tools import transfer_to_stats_agent
from tools import WebSearch

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
            "Dont not generate any Network Analysis of Authors and Affiliations or Author Network Graphs. These are out of the scope\n\n!"
            "Come up with at least 5 ideas before you transfer the user to the stats agent. Thank you!!! \n\n"
        ),
        name="ds_assistant",
    )
    return research_ds_agent