from langgraph_swarm import create_swarm
from classes import State
from agents import Research_DataScience_Agent
from agents import Research_Stat_Agent

def Swarm_Agent():
    research_ds_agent = Research_DataScience_Agent(State)
    print('\n\nCreated research_ds_agent\n\n')
    research_stat_agent = Research_Stat_Agent(State)
    print('\n\n Created research_stat_agent \n\n')
    swarm_agent = create_swarm(
    agents=[research_ds_agent, research_stat_agent],
    default_active_agent = 'ds_assistant',
    )
    return swarm_agent.compile()

