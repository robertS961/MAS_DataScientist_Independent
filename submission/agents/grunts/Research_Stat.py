from classes import State
from langgraph.prebuilt import create_react_agent
from tools.Web_Search import WebSearch
from tools.Transfer_To_Ds_Agent import transfer_to_ds_agent
from tools import scrape_webpages



def Research_Stat_Agent(state:State):
    web_search = WebSearch()
    ds_agent = transfer_to_ds_agent()
    previous_message = state['messages']
    # previous_feedback = state['feedback']
    data = state['dataset_info']
    #vis_agent = transfer_to_vis_agent()
    research_stat_agent = create_react_agent(
        model="openai:gpt-4o",
        tools=[web_search],
        prompt=(
            "You are a research statistician agent.\n\n"
            "INSTRUCTIONS:\n"
            "Your goal is find novel statistical learning ideas for a tabular dataset."
            f"You might have a previous messaged from another agent or feedback from other agents. It will be in these messages : {previous_message}\n\n"
            "If there is feedback please make sure to improve the previous messages with it! \n\n"      
            f"These are the columns of the tabular data. Use them in the ideas by citing the column names: {data} \n\n"
            "You have access to two different tools! \n\n"
            "You can web_search the internet to discover ideas! \n\n"
            "You can also call on the ds (Data science agent) \n\n"
            "Feel free to discover interesting ideas on your own, add to the ideas of the previous message, and search the web! \n\n!"
            "Dont need come up with any Network Analysis of Authors and Affiliations or Network Graphs for Authors! \n\n"
            "If there is feedback make sure you use this to improve your results espeically adding onto other ideas and creating new ideas! \n\n"
            "Return all the ideas in a list. Make sure there is at least 6 ideas then transfer the user to the ds(data science) agent\n\n"
            "Thank you! \n\n"
        ),
        name="stats_assistant",
    )
    return research_stat_agent