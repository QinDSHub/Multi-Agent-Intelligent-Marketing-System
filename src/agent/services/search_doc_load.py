from langchain_community.document_loaders import WebBaseLoader
from pydantic import BaseModel, Field
from typing import List
from agent.services.search_agent import AllSearchResults


class SearchDocResult(BaseModel):
    title: str = Field(..., description="The title of the search result")
    content: str = Field(..., description="The main body content re-edited from the crawled webpage by llm")


class AllSearchDocResults(BaseModel):
    results: List[SearchDocResult] = Field(default_factory=list)


from llm_model import _make_llm
from langchain_core.prompts import ChatPromptTemplate

def text_loader(search_results: AllSearchResults) -> AllSearchDocResults:
     
    output_results = AllSearchDocResults()
    for result in search_results.results:
        link = result.link
        title = result.title
        # description = result.snippet

        loader = WebBaseLoader(link)
        docs = loader.load()
        res = docs[0].page_content
        clean_chunk = []
        for chunk in res.split("\n"):
            if chunk=='':
                    continue
            else:
                    chunk = chunk.strip()
                    if len(chunk.split(' ')) < 10:
                        continue
                    clean_chunk.append(chunk)
        final_content = "\n\n".join(clean_chunk)

        strict_llm = _make_llm("gpt-5-nano", 0.2)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Senior Copy Editor specializing in proofreading the text content."),
            ("human", "The following is raw text extracted from a website. Please filter out noise such as navigation bars, button text, and dates to preserve only the primary body content.\n\nRaw Content: {raw_content}")
        ])
        chain = prompt | strict_llm
        cleaned_content = chain.invoke({"raw_content": final_content})
        output_results.results.append(SearchDocResult(title=title, content=cleaned_content.content))

    return output_results


