import shutil
import os
from typing import List
from pydantic import BaseModel, Field

from agent.services.search_doc_load import AllSearchDocResults 
from agent.services.local_doc_load import AllLocalDocResults
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from llm_model import _embed_model

class RagResult(BaseModel):
    content: List[str] = Field(..., description="The similar content list")

def map_agent(input_docs: AllSearchDocResults | AllLocalDocResults, db_path: str) -> Chroma:
    langchain_docs = [
        Document(page_content=res.content, metadata={"title": res.title}) 
        for res in input_docs.results
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(langchain_docs)
    
    embedding_func = _embed_model(model='text-embedding-ada-002')
    
    return Chroma.from_documents(
        documents=split_docs, 
        embedding=embedding_func, 
        collection_name="openai_embedding", 
        persist_directory=db_path,
        collection_metadata={"hnsw:space": "cosine"}
    )

def search_with_threshold(vectorstore, query, threshold=0.5):

    results_with_scores = vectorstore.similarity_search_with_relevance_scores(query, k=50)
    return [doc for doc, score in results_with_scores if score >= threshold]

def reduce_agent(input_docs_1: AllSearchDocResults, input_docs_2: AllLocalDocResults, db_path: str) -> RagResult:

    if not input_docs_1.results and not input_docs_2.results:
        return RagResult(content=[])
        
    query = input_docs_1.results[0].query if input_docs_1.results else "default query"

    if os.path.exists(db_path):
        try:
            shutil.rmtree(db_path)
        except PermissionError:
            print(f"Warning: Directory {db_path} is in use, attempting to continue...")

    map_agent(input_docs_1, db_path)
    map_agent(input_docs_2, db_path)

    vectorstore = Chroma(
        persist_directory=db_path, 
        embedding_function=_embed_model(model='text-embedding-ada-002'),
        collection_name="openai_embedding"
    )
    
    docs = search_with_threshold(vectorstore, query, threshold=0.5)
    return RagResult(content=[d.page_content for d in docs])

      
