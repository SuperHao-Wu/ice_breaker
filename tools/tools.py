from langchain.agents import tool
from langchain.serpapi import SerpAPIWrapper


@tool()
def get_profile_url(text: str) -> str:
    """Searches for linkedin Profile Page"""
    search = SerpAPIWrapper(search_engine="test")
    res = search.run(f"{text}")

    return res
