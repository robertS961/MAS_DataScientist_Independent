
from langgraph.prebuilt import create_react_agent
from classes import State

def reducer_agent(state: State, ideas:int = 5):
    data = state['dataset_info']
    message = state['messages']
    prompt = (
        "You are a machine learning expert with over 25 years of experience! \n" 
        "You are given a list of several data science, statistical learning, and machine learning ideas \n"
        f"They are to applied to the tabular data of these columns \n {data} \n"
        f"The ideas are listed here \n {message} \n !"
        #"Remove any ideas that would take a long run time \n"
        "Remove any ideas that are Network Analysis of Authors and Affiliations or Graph creation of Author. Anything like that ! \n"
        "Remove any ideas that are duplicates in the list or extremely similar! \n"
        #"Remove any ideas that are too complex, we want fast run time!"
        f"Then take the best {ideas} ideas from the remaining ones! \n"
        f"Return the those ideas in the same format! \n"
    )
    #websearch = WebSearch()
    reducer_agent = create_react_agent(
        model="openai:gpt-4o", 
        tools=[],
        name = "reducer_agent",
        prompt = prompt
    )
    return reducer_agent