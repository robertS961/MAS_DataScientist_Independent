from langgraph.prebuilt import create_react_agent
from classes import State
from helper_functions import get_llm

def narrative_agent(state: State):
    llm = get_llm(model = 'gpt-5-mini')
    code = state['code']
    prompt = (
        f"You will be given python plotly code to be executed on a tabular data set. It will display graphs on various statistical, machine learning, and data science ideas. \n"
        "Your goal is to take this code for an HTML server and make it a cohesive report for a HTML website\n"
        "If each section has a narrative then enhance it based on what the graph is stating and why it would be important to the reader. This should be displayed on the HTML website within a tag or whatever makes the most sense.  \n"
        "If the graph doesn't have a narrative then write one, once again inside the correct HTML tag. The goal is to explain what the graph is doing, why its important, and the significance of the findings and why these findings are important \n"
        "Make sure each pharagraph flows into the next one just like an actual report \n"
        "Lastly add an introduction to the entire flow and a conclusion. Make sure the introduction goes over the purpose of the paper and the flow of what the readers will read\see. Make sure the concluscion wraps the entire paper up with key points from the visualizations and possible next steps for research scientists.\n"
        "Make all the writing extremely educational, diving into the purposes of the graphs and algorithms but not making it overly complicated. There is power in simplicity!"
        "To concluded do not touch the python code, add all of these changes to HTML/Javascript Code , only by adding narratives, don't change the meat of that code either!\n"
        "Return all the python code and the newly edited javascript/html code bewteen ```python #CODE HERE ```\n"
        "Make sure there isnt any bugs with the new additions. Thank you!"
    )
    #websearch = WebSearch()
    narrative_a = create_react_agent(
        model=llm, # Change it back to "openai:o4-mini",
        tools=[],
        name = "Narrative_Agent",
        prompt = prompt
    )
    return narrative_a