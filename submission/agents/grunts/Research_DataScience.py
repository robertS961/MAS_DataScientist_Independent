from classes import State, globe
from langgraph.prebuilt import create_react_agent
from tools import transfer_to_stats_agent
from tools import WebSearch
from tools import scrape_webpages
from helper_functions import get_llm, update_edge

def Research_DataScience_Agent(state:State):
    llm = get_llm(model = 'gpt-5-mini')
    web_search = WebSearch()
    stats_agent = transfer_to_stats_agent()
    print(f"\n\nThis is the current message {state['messages']}\n\n")
    previous_message = state['messages']
    # previous_feedback = state['feedback']
    data = state['dataset_info']
    research_ds_agent = create_react_agent(
        model=llm,
        tools=[web_search],
        prompt=(
            "You are a research data science agent.\n\n"
            "INSTRUCTIONS:\n"
            "Your goal is find novel data science ideas for a tabular dataset. These ideas should be able to be applied to tabular data .\n\n"
            f"You might have a previous messages or feedback from another agents it is: {previous_message}"
            "IF there is feedback in the aboves messages please apply it to our previous state of research ideas if they have been created! \n\n"
            f"These are the columns of the tabular data. Use them in the ideas by citing the column names: {data} \n\n"
            "You have access to two different tools! \n\n"
            "You can web_search the internet to discover ideas! \n\n"
            "You can also transfer you knowledge to the statistican agent \n\n"
            "Once you discover interesting and novel ideas then list them! Make sure you use the feedback in the ideas if there is any! \n\n"
            "Dont not generate any Network Analysis of Authors and Affiliations or Author Network Graphs. These are out of the scope\n\n!"
            "Come up with at least 6 ideas before you transfer the user to the stats agent. Thank you!!! \n\n"
        ),
        name="ds_assistant",
    )
    return research_ds_agent