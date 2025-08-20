from langgraph.prebuilt import create_react_agent
from classes import State
from helper_functions import get_llm

def narrative_agent(state: State):
    llm = get_llm(model = 'gpt-5-mini')
    code = state['code']
    prompt = (
        f"You will be given python plotly code to be executed on a tabular data set. It will display graphs on various statistical, machine learning, and data science ideas. \n"
        "Your goal is to take this code for an HTML server and make it a cohesive report for a HTML website\n"
        "Add a paragraph to each graph along with an introduction and conclusion, each needs to be at least a paragraph long! \n"
        "Each paragraph should explain the algorithms being used and why this algorithm was selected! \n"
        "The paragraph should also explain what the graph is doing, why its important, What do the axis and keys mean, and the significance of the findings and why these findings are important \n"
        "We want also each paragraph to explain what significant findings would look like in this graph, or what is considered average. This way the user can easily interupt the graph without any expertise! \n"
        "Make sure each paragraph flows into the next one just like an actual report and it is okay if you need more than one paragraph per graph to achieve these goals!\n"
        "Make sure the introduction goes over the purpose of the paper and the flow of what the readers will read\see. Make sure the conclusion wraps the entire paper up with key points from the visualizations and possible next steps for research scientists.\n"
        "Make sure the paper has a title, each section has a title, and the introduction and conclusion are clearly labled with titles too.\n "
        "All of these paragraphs should be interserted int the HTML with <p> or whatever makes sense. Make the font large enough to easily be able to read! \n"
        "Make all the writing extremely educational and that someone without any expertise in this field could understand! We want an extremely education and easy to read report!\n"
        "Lastly highlight and bold the main take aways from each graph in the paragraph(s). This should be 1-3sentences that tell the big picture! Make sure the correct tags are included in the HTML for this to work!\n"
        "To conclude do not touch the python code, add all of these changes to HTML/Javascript Code , only by adding narratives, don't change the meat of that code either!\n"
        "Return all the python code and the newly edited javascript/html code bewteen ```python #CODE HERE ```\n"
        "Make sure there isnt any bugs with the new additions. Thank you!\n"
    )
    #websearch = WebSearch()
    narrative_a = create_react_agent(
        model=llm, # Change it back to "openai:o4-mini",
        tools=[],
        name = "Narrative_Agent",
        prompt = prompt
    )
    return narrative_a