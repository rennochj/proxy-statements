import os

from chromadb import PersistentClient
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API key and Azure endpoint must be set")

embeddings = OpenAIEmbeddings(api_key=SecretStr(api_key))

client = PersistentClient(path="dbstore/")

vectordb = Chroma(
    client=client,
    # collection_name="OLLAMA"
)

print(client.list_collections())

retriever = vectordb.as_retriever()

print(vectordb.get().keys())
print(vectordb.get()["metadatas"])
