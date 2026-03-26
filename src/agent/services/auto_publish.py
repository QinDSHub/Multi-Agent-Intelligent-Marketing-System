from typing import List
from pydantic import BaseModel, Field
import requests
from agent.services.content_generation import AllMarketingContents

class FacebookPostRequest(BaseModel):
    marketing_data: AllMarketingContents
    page_id: str = Field(..., description="FB Page ID")
    access_token: str = Field(..., description="FB Page Access Token")

class SinglePostResult(BaseModel):
    content_id: str
    post_id: str = ""
    status: str
    response: dict

class DistributorOutput(BaseModel):
    results: List[SinglePostResult]


def distributor_agent(request: FacebookPostRequest) -> DistributorOutput:
    final_results = []
    
    url = f"https://graph.facebook.com/v17.0/{request.page_id}/feed"

    for item in request.marketing_data.contents:

        if "Poster" in item.content_format or "Facebook" in str(item.distribution_channel):
            
            message = f"{item.headline}\n\n{item.body_text}\n\n{item.call_to_action}"

            payload = {
                "message": message,
                "access_token": request.access_token
            }

            try:
                resp = requests.post(url, data=payload, timeout=15)
                resp.raise_for_status()
                api_res = resp.json()
                
                final_results.append(SinglePostResult(
                    content_id=item.insight_id,
                    post_id=api_res.get("id", ""),
                    status="success",
                    response=api_res
                ))
            except Exception as e:
                final_results.append(SinglePostResult(
                    content_id=item.insight_id,
                    status="failed",
                    response={"error": str(e)}
                ))
                
    return DistributorOutput(results=final_results)