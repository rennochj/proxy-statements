import os
from openai import AzureOpenAI

api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

if not api_key or not azure_endpoint:
    raise ValueError("API key and Azure endpoint must be set")

client = AzureOpenAI(
  api_key = api_key,  
  api_version = "2023-07-01-preview",
  azure_endpoint = azure_endpoint
)

response = client.chat.completions.create(
    model="gpt-35-turbo-16k",
    messages=[
        {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
        {"role": "user", "content": "Who were the founders of Microsoft?"}
    ]
)

