from classes import State, Configurable
from agents import supervisor, create_judge
from helper_functions import create_reflection_graph

def create_research_team(state:State):
    supervis = supervisor(State, name = "research_supervisor")
    #print('supervisor created \n')
    #judge = create_judge()
    #print('Judge is created \n')
    #research_graph = create_reflection_graph(supervis, judge, State, Configurable)
    return supervis



