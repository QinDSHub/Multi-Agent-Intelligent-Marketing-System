from typing import List, Optional
from pydantic import BaseModel, Field
from serpapi import GoogleSearch
from env_utils import SERPAPI_API_KEY

class SearchResult(BaseModel):
    query: str = Field(..., description="The search query")
    link: str = Field(..., description="The URL of the search result")
    title: str = Field(..., description="The title of the search result")
    source: str = Field(..., description="The source of the search result")
    date: Optional[str] = Field(None, description="The publication date")
    snippet: str = Field(..., description="A brief summary")

class AllSearchResults(BaseModel):
    results: List[SearchResult] = Field(default_factory=list)

def google_search(query: str) -> AllSearchResults:

    all_results = AllSearchResults()
    
    search = GoogleSearch({
        "q": query,
        "location": "Dublin, Ireland",
        "gl": "IE",
        "hl": "en",
        "tbm": "nws",
        "num": 10,
        "api_key": SERPAPI_API_KEY
    })
    
    res = search.get_dict()
    news_items = res.get("news_results", [])

    if not news_items:
        print(f"No news results found for the query: {query}, pls double check or try later!")
        return all_results

    for item in news_items:
        result_obj = SearchResult(
            query = query,
            link=item.get("link", ""),
            title=item.get("title", ""),
            source=item.get("source", ""),
            date=item.get("date", ""),
            snippet=item.get("snippet", "")
        )
        all_results.results.append(result_obj)
        
        print(f"Title: {result_obj.title} | Source: {result_obj.source} | Link: {result_obj.link}")

    return all_results
