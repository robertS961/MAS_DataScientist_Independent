
from langgraph_swarm import create_handoff_tool

def transfer_to_stats_agent():
    transfer_to_stats_agent = create_handoff_tool(
        agent_name = 'stats_assistant',
        description = (
            'Transfer user to a statistician agent! Capabale of coming up with novel statistical learning ideas \n\n'
            'It uses a websearch tool Tavily to help find research ideas for statisticial learning and statistics \n\n'
        ),
    )
    return transfer_to_stats_agent