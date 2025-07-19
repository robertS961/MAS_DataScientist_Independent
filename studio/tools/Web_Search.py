from langchain_tavily import TavilySearch

def WebSearch():
    """ This function searches the web for relevant research information! """
    web_search = TavilySearch(max_results=1)
    return web_search