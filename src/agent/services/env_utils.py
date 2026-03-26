import dotenv
import os

dotenv.load_dotenv(override = True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_BASE_URL = os.getenv("TAVILY_BASE_URL")

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")



