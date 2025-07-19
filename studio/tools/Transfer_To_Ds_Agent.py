from langgraph_swarm import create_handoff_tool

def transfer_to_ds_agent():
    transfer_to_ds_agent = create_handoff_tool(
        agent_name = 'ds_assistant',
        description = (
            'Transfer user to a data science agent! Capabale of coming up with novel data science ideas \n\n'
            'It uses a websearch tool Tavily to help find research ideas for data science \n\n'
        ),
    )
    return transfer_to_ds_agent