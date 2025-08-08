
from langchain_tavily import TavilySearch


def WebSearch():
    """ This function searches the web for relevant research information! """
    web_search = TavilySearch(max_results=1, tavily_api_key= "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")
    return web_search