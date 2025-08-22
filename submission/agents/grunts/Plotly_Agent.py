
from langgraph.prebuilt import create_react_agent
from classes import State
from helper_functions import get_llm

def ploty_agent(state: State):
    llm = get_llm(model = 'gpt-5-mini')
    prompt = (
        "You are a python plotly expert with over 10 years of experience! \n"
        "You will be given ideas to be executed on a tabular data set. It will be ideas on various statistical, machine learning, and data science ideas. \n"
        "Take these ideas and turn them into interactive and beautiful plotly graphs! Make sure the display is in dark mode!\n"
        "Make sure each plotly graph is large enough to clearly understand. It is fine if the user need to scroll left/right and up/down! If the axis have lots of words, make sure to space the graphs out! \n"
        "Make the colors of the graphs vibrant and clear, avoid using dull and similar colors! This applies to any buttons, make sure the button is a different color than the background and the button writing is also a different color!  \n"
        "When displaying the graphs please keep the following format : Main title, Sub titles(if any), PLENTY OF space, then interactive buttons(keys,toggles,) make sure to give them space between them, then space, then graph \n"
        "Make sure the hover/onselect for buttons is a different color than white. There is a ton of WHITE ON WHITE, do NOT WANT THIS! \n"
        "Make sure the toggle buttons for the legend or key isn't on top of the heat map. Instead move the heat map futher over or move the key/legend to the bottom of the graph.\n"
        "When updating the layout or titles make sure to include 'text' like this {'title': {'text': 'Novelty vs Citations-per-Year — Binned averages'}} for plotly, as just 'title' will delete the previous title. NOT JUST 'title'\n"
        "When using '.add_annotations'  make sure you give the button enough space. It normally appears TOO CLOSE! \n"
        "Make sure each graph has a narrative description of what is going on in the graph and how to interprut it \n"
        "Once Plotly Code is written add Html/Javascript code to display all the figures so it can be ran on a server!\n"
        "For the HTML/Javascript code, make sure to include the Plotly JS via CDN directly inside each div for better compatibility! Thus include_plotlyjs='cdn' for each div, NOT FALSE(This is outdated!) \n"
        "Do NOT use '<script src=https://cdn.plot.ly/plotly-latest.min.js></script>' in the HTML code, instead use include_plotlyjs='cdn' in the plot function! \n"
        "Make sure the code runs without errors and bugs! Return the code like ```python #CODE HERE ```\n"
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