from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from llm_model import _make_llm_with_structure
from agent.services.insights_extract import AllStrategicInsights

class MarketingContent(BaseModel):
    insight_id: str = Field(..., description="Matches the ID from the source insight")
    content_format: str = Field(..., description="e.g., Blog, Articles, Poster, Video Script, Study Case")
    headline: str = Field(..., description="Catchy headline for the content")
    body_text: str = Field(..., description="The full copy of the marketing content")
    call_to_action: str = Field(..., description="Clear CTA for the reader")
    target_audience: str = Field(..., description="Primary persona this content targets")
    distribution_channel: List[str] = Field(..., description="Recommended channels, e.g., ['LinkedIn', 'Twitter']")
    sources: List[str] = Field(..., description="Reference IDs or links used")

class AllMarketingContents(BaseModel):
    contents: List[MarketingContent]

content_llm = _make_llm_with_structure(AllMarketingContents, "gpt-5-nano", 0.7)

CONTENT_PROMPT = ChatPromptTemplate.from_template("""
    You are an expert Marketing Strategist. 
    Generate concrete, ready-to-publish marketing content based on the following strategic insights.

    For each insight:
    1. Select the most effective content format for the target audience.
    2. Write professional, high-converting copy including a headline and body text.
    3. Ensure a compelling Call-to-Action (CTA).

    Strategic Insights to process:
    {insights}
    """)

def marketing_content_agent(insights: AllStrategicInsights) -> AllMarketingContents:

    formatted_insights = "\n".join(
        f"ID: {i.insight_id} | Insight: {i.key_insight_content} | Relevance: {i.strategic_relevance}"
        for i in insights.insights
    )

    chain = CONTENT_PROMPT | content_llm

    res = chain.invoke({"insights": formatted_insights})

    if isinstance(res, AllMarketingContents):
        return res
    return AllMarketingContents(**res) if isinstance(res, dict) else res