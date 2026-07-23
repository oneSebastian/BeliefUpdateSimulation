import os
import requests
from dotenv import load_dotenv

load_dotenv() # loads .env from current working directory


class AcademicAIClient:
    def __init__(self, model, base_url: str = "https://it-u-api.academic-ai.at"):
        self.api_key = os.environ["ACADEMIC_AI_ID"]
        self.api_secret = os.environ["ACADEMIC_AI_SECRET"]
        self.base_url = base_url
        model_map = {
            "academic-ai-gpt-5-mini": "gpt-5-mini",
        }
        self.model = model_map.get(model, model)

    def create_chat_completion(self, messages: list, **kwargs):
        url = f"{self.base_url}/api/v1/llm/chat"
        headers = {
            "X-Client-ID": self.api_key,
            "X-Client-Secret": self.api_secret,
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            **kwargs,
        }
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()

