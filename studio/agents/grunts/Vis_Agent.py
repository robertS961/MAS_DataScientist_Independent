from classes import State
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver


def vis_a(state:State):
    tools = []
    pass_msg = False
    data = state['dataset_info']
    code = state['messages'][-1].content 
    error = state['error']
    if state['flag']: 
        state['ideas'] = state['messages'][-1].content
        state['flag'] = False
    ideas = state['ideas']

    if error == "":
        prompt = (
            "You are a Python visualization expert. You generate stunning visualizations using matplotlib, seaborn, plotly, or any libraries you can think of!.\n\n"
            "We have already created several ideas. Your objective is to take the ideas and code them ! \n\n"
            f"The dataset columns are as follows: {data}\n\n"
            f"The ideas are as follows: {ideas}\n\n"
            "Respond ONLY with Python code that creates meaningful and aesthetic data visualizations. Do not explain anything. Thank you \n\n"
            "Make sure the number of plots you create matches the number of ideas, if it is impossible create more data science ideas from the columns! \n\n"
            "Make sure the Python code runs and there isn't any bugs. Also write it in simple python code so the users can easily understand it \n\n"
            "Also Sklearn is depreciated use scikit-learn! \n\n"
        )
    else:
        prompt = (
            "You are a Python visualization expert. You generate stunning visualizations using matplotlib, seaborn, plotly, or any libraries you can think of!.\n\n"
            f"Code has already been generated for these several ideas {ideas} \n\n "
            f"The following code was created for those above ideas{code} based on these data columns {data}\n\n"
            f"However there were errors in the code. Here is the error {error}\n\n"
            "Please fix the code and improve the clariy of any graphs that seems confusing, or axis that have long labels/uncertain, make the coloring more clear, etc.\n\n"
            "Respond ONLY with Python code that creates meaningful and aesthetic data visualizations. Do not explain anything. Thank you \n\n"
            "Make sure all the ideas are still coded for! Do not cut ideas because the code doesn't work. Instead fix the code slightly so the idea still works! \n\n"
            "Make sure the number of plots you create matches the number of ideas, if it is impossible then create more data science like plots from the columns! \n\n"
            "Make sure the Python code runs and there isn't any bugs. Also write it in simple python code so the users can easily understand it \n\n"
            "Also Sklearn is depreciated use scikit-learn! \n\n"
        )
    vis_agent = create_react_agent(
        model = "openai:gpt-4o",
        tools = [],
        prompt = prompt,
        name = 'vis_agent',
        checkpointer=InMemorySaver()
    )
    return vis_agent