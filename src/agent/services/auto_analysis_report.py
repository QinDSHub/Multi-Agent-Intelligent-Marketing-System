from typing import List, Optional
from pydantic import BaseModel, Field
import requests

class PostPerformance(BaseModel):
    post_id: str
    reach: int = Field(default=0, description="Unique users who saw the post")
    impressions: int = Field(default=0, description="Total times the post was viewed")
    clicks: int = Field(default=0, description="Total link clicks")
    engagement: int = Field(default=0, description="Likes, comments, and shares")
    ctr: float = Field(default=0.0, description="Click-through rate (clicks/impressions)")

class AnalyticsReport(BaseModel):
    summary_report: List[PostPerformance]
    total_avg_ctr: float
    top_performing_post_id: Optional[str]


def analytics_agent(post_ids: List[str], access_token: str) -> AnalyticsReport:

    report_data = []
    
    metrics = "post_impressions_unique,post_impressions,post_clicks_by_type,post_engagements"
    
    for pid in post_ids:

        url = f"https://graph.facebook.com/v17.0/{pid}/insights"
        params = {
            "metric": metrics,
            "access_token": access_token
        }
        
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            
            stats = {item["name"]: item["values"][0]["value"] for item in data}
            
            clicks = stats.get("post_clicks_by_type", {}).get("link click", 0) 
            if isinstance(clicks, dict): clicks = sum(clicks.values()) 
            
            impressions = stats.get("post_impressions", 0)
            ctr = (clicks / impressions) if impressions > 0 else 0.0
            
            report_data.append(PostPerformance(
                post_id=pid,
                reach=stats.get("post_impressions_unique", 0),
                impressions=impressions,
                clicks=clicks,
                engagement=stats.get("post_engagements", 0),
                ctr=round(ctr, 4)
            ))
            
        except Exception as e:
            print(f"Could not capture data for post {pid}: {e}")

    if not report_data:
        return AnalyticsReport(summary_report=[], total_avg_ctr=0.0)

    avg_ctr = sum(p.ctr for p in report_data) / len(report_data)
    top_post = max(report_data, key=lambda x: x.clicks).post_id if report_data else None

    return AnalyticsReport(
        summary_report=report_data,
        total_avg_ctr=round(avg_ctr, 4),
        top_performing_post_id=top_post
    )

