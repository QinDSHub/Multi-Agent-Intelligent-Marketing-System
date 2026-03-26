from agent.services.rag_agent import RagResult
from llm_model import _make_llm_with_structure
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

class StrategicInsight(BaseModel):
    insight_id: str = Field(..., description="Unique identifier for the insight")
    key_insight_content: str = Field(..., description="The main content of the key insight")
    strategic_relevance: str = Field(..., description="Explanation of why this insight is strategically relevant to the business")
    
class AllStrategicInsights(BaseModel):
    insights: List[StrategicInsight]

_llm_structure = _make_llm_with_structure(AllStrategicInsights, "gpt-5-nano",0.7)

INSIGHTS_PROMPT = ChatPromptTemplate.from_template("""
    Merge, deduplicate and synthesize the following insights
    into FIVE most important and strategic insights.

    Insights:
    {insights}

    Output JSON strictly matching:
    {schema}
    """, partial_variables={"schema": AllStrategicInsights.model_json_schema()})


def insights_agent(raw_insights: RagResult) -> AllStrategicInsights:
    raw_insights = RagResult(**raw_insights.model_dump())
    chain = INSIGHTS_PROMPT | _llm_structure
    res = chain.invoke({"insights": "\n".join(raw_insights.content)})
    return AllStrategicInsights(**res)