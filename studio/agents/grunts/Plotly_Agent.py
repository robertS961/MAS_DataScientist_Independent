
from langgraph.prebuilt import create_react_agent
from classes import State
from helper_functions import get_llm

def ploty_agent(state: State):
    code_type = state['code_type']
    llm = get_llm(model = 'o4-mini') # Lets make this model us chat gpt 4o mini
    prompt = (
        f"You are a {code_type} expert with over 10 years of experience! \n"
        "You will be given ideas to be executed on a tabular data set. It will be ideas on various statistical, machine learning, and data science ideas. \n"
        f"Take these ideas and turn them into interactive and beautiful {code_type} graphs!\n"
        "Make sure to include a narrative description of the graphs and what they represent! Make it interactive too, maybe a button press to show the narrative description!\n"
        "Make the colors of the graphs vibrant and clear, avoid using dull and similar colors! \n"
        f"Once {code_type} Code is written add Html/Javascript code to display all the figures so it can be ran on a server!\n" if code_type == 'plotly' else "Once vega-lite Code is written, double check the code and make sure it is correct and runs without errors! \n"
        f"For the HTML/Javascript code, make sure to include the Plotly JS via CDN directly inside each div for better compatibility! Thus include_plotlyjs='cdn' for each div, NOT FALSE(This is outdated!) \n" if code_type == 'plotly' else ""
        f"Do NOT use '<script src=https://cdn.plot.ly/plotly-latest.min.js></script>' in the HTML code, instead use include_plotlyjs='cdn' in the plot function! \n" if code_type == 'plotly' else ""
        "Make sure the code runs without errors and bugs! \n"
        "Thank you! \n"
    )
    #websearch = WebSearch()
    plotly_ = create_react_agent(
        model= llm, # Change it back to "openai:o4-mini",
        tools=[],
        name = "Plotly_Agent",
        prompt = prompt
    )
    return plotly_