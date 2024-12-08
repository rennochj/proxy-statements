import os
from openai import AzureOpenAI
from pypdf import PdfReader
import pandas as pd
from tenacity import retry, wait_random_exponential

api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

if not api_key or not azure_endpoint:
    raise ValueError("API key and Azure endpoint must be set")

client = AzureOpenAI(
  api_key = api_key,  
  api_version = "2023-07-01-preview",
  azure_endpoint = azure_endpoint
)

# @retry(wait=wait_random_exponential(multiplier=1, max=10)) #retry decorator to get around rate limits
def get_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return response
 
 
def get_embedding_from_pdf(pdf):
    file = PdfReader(pdf)
    text = ' '.join([page.extract_text() for page in file.pages])
    return get_embedding(text)
 
 
# list of pdf paths in "PDFs/" subdirectory
import os
df = pd.DataFrame({'pdf': [pdf for pdf in sorted(os.listdir('../test_statements/')) if pdf.endswith('.pdf')]})

print(df)

df['embedding'] = df['pdf'].apply(lambda x: get_embedding_from_pdf('../test_statements/'+x)) # type: ignore

print(df)

