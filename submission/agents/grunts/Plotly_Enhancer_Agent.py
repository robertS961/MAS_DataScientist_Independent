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
        f"Also fix the axis ranges to ensure that the graphs display meaningful data and are not blank!  Use this information about the data \n {state['data_description']}\n"
        f"Use this information also {state['data_info']}, make sure the code is prepared for NaN values, duplicate values, and missing values! \n"
        "Also some of the graph displays are not clear, based on the data, like for line plots there might be doubles of the same pair. Make sure to aggregate them together for line plots, bar plots, etc. to display clear figures! \n"
        "Make sure the graphs have keys and they are easy to understand! \n"
        "Make sure the narrative descriptions of the graphs explain the graph and the importance why this is being graphed and the explination of any tool used(like machine learning or algorithms)! \n"
        "Once the enhancements are made, add HTML/Javascript code at the end to display all the figures so it can be run on online servers! \n"
        "For the HTML/Javascript code, make sure to include the Plotly JS via CDN directly inside each div for better compatibility! Thus include_plotlyjs='cdn', NOT FALSE(This is outdated!) \n"
        "Do NOT use '<script src=https://cdn.plot.ly/plotly-latest.min.js></script>' in the HTML code, instead use include_plotlyjs='cdn' in the plot function! \n"
        "Make sure the code runs without errors and bugs! Return it like ```python #CODE HERE ```\n"
        "Return the full python code with plotly and the html/javascript code at the end without the use of the script above and having include_plotlyjs='cdn' in each div ! \n "
        "Thank you! \n"
    )
    '''
    prompt = (
        "You are a python plotly expert with over 10 years of experience! \n"
        "You will be given code to be executed on a tabular data set. It will display graphs on various statistical, machine learning, and data science ideas. \n" 
        "You will take this python code and turn it into interactive and beautiful plotly graphs! \n"
        "Keep the same graphs and displays, just turn them into plotly graphs \n"
        "Then add HTML/Javascript Code at the end to display all the figures and print it to output.html file so it can be run on online servers! \n"
        "There could be double pairs of columns that are referenced in the code, so make sure Aggregate them together for line plots,bar plots, etc. to display clear figures! \n"
        "Please respond with the full python code with plotly and the html/javascript code at the end! \n"
        "Make sure the code runs without errors and bugs! \n"
        "Thank you! \n"
    )
    '''
    #websearch = WebSearch()
    plotly_enhancer = create_react_agent(
        model=llm, # Change it back to "openai:o4-mini",
        tools=[],
        name = "Plotly_Enhancer_Agent",
        prompt = prompt
    )
    return plotly_enhancer