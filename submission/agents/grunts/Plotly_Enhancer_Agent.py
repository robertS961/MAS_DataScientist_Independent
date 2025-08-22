from langgraph.prebuilt import create_react_agent
from classes import State
from helper_functions import get_llm

def ploty_enhancer_agent(state: State):
    llm = get_llm(model = 'gpt-5-mini')
    prompt = (
        "You are a python plotly expert with over 20 years of experience! \n"
        "You will be given python plotly code to be executed on a tabular data set. It will display graphs on various statistical, machine learning, and data science ideas. \n"
        "These graphs have been created by a previous agent, but they need to be enhanced for better clarity and interactivity. \n"
        "Your task is to enhance the existing plotly code by improving the visualizations, adding interactivity, and ensuring that the graphs are clear and informative. Make sure the colors are not too similar as it is difficult to see! Make it also in dark mode! \n"
        "Make sure the graphs are large enough to clearly see the information. No small graphs please!Maybe make them all the same size and make sure there is enough spacing between them for wordy axis and labels! \n"
        f"Also fix the axis ranges to ensure that the graphs display meaningful data and are not blank!  Use this information about the data \n {state['data_description']}\n"
        f"Use this information also {state['data_info']}, make sure the code is prepared for NaN values, duplicate values, and missing values! \n"
        "Also some of the graph displays are not clear, based on the data, like for line plots there might be doubles of the same pair. Make sure to aggregate them together for line plots, bar plots, etc. to display clear figures! \n"
        "Make sure each graph has a narrative of what the graph is doing and how to find meaningful results from it along with what the algorithm does and why its important!\n"
        "Make sure the graphs have keys and they are easy to understand! Also make sure if the graph has anythig interactive (buttons, sliders, etc.) that they don't overlap with the title, axis, or graph! Also make sure these buttons and button writing have different colors from each other and the background along with the hover feature not whiting out the button! \n"
        "Make sure on the double graphs where they are displayed next to each other that there is plenty of space between. Also on single graphs or double there is occasionaly keys or right side bar/key is on top of the graph or intersecting with another graph. Give the graph keys space along with the graphs heatmap , they usually overlap and it is IMPOSSIBLE to click on the toggle keys. Don't worry about room as there is plenty of scroll room to the right or left! \n"
        "The button hovers and selected button are normally unreadable its usually white on white. Change this so the hover or selected button is light green!\n"
        "When updating the layout or titles make sure to include 'text' like this {'title': {'text': 'Novelty vs Citations-per-Year — Binned averages'}} for plotly, as just 'title' will delete the previous title. NOT JUST 'title'\n"
        "When using '.add_annotations'  make sure you give the button enough space. It normally appears TOO CLOSE! \n"
        "Once the enhancements are made, add HTML/Javascript code at the end to display all the figures so it can be run on online servers! \n"
        "For the HTML/Javascript code, make sure to include the Plotly JS via CDN directly inside each div for better compatibility! Thus include_plotlyjs='cdn', NOT FALSE(This is outdated!) \n"
        "Do NOT use '<script src=https://cdn.plot.ly/plotly-latest.min.js></script>' in the HTML code, instead use include_plotlyjs='cdn' in the plot function! \n"
        "Make sure the code runs without errors and bugs! Return it like ```python #CODE HERE ```\n"
        "Return the full python code with plotly and the html/javascript code at the end without the use of the script above and having include_plotlyjs='cdn' in each div ! \n "
        "Thank you! \n"
    )
    #websearch = WebSearch()
    plotly_enhancer = create_react_agent(
        model=llm, # Change it back to "openai:o4-mini",
        tools=[],
        name = "Plotly_Enhancer_Agent",
        prompt = prompt
    )
    return plotly_enhancer