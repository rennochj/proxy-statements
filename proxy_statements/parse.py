import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
from pydantic import SecretStr

pdf_folder_path = "../test_statements/"
documents = []

for file in os.listdir(pdf_folder_path):
    if file.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder_path, file)
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
        
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
chunked_documents = text_splitter.split_documents(documents)

client = chromadb.PersistentClient(path="dbstore/")

print(f"collections - {client.list_collections()}")
if not "consent_collection" in [ c.name for c in client.list_collections()]:
        print("Collection created")
        consent_collection = client.create_collection("consent_collection")
else:
    print("Collection recreated")
    client.delete_collection("consent_collection")
    consent_collection = client.create_collection("consent_collection")

print(f"collections - {client.list_collections()}")

api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

if not api_key or not azure_endpoint:
    raise ValueError("API key and Azure endpoint must be set")

openai_ef = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=SecretStr(api_key),
    azure_endpoint=azure_endpoint
)

vector_db = Chroma.from_documents(
    documents=chunked_documents,
    embedding=openai_ef,
    persist_directory="dbstore/"
)

