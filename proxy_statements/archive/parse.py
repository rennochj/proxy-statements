import os
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
from pydantic import SecretStr

COLLECTION_NAME = "OPENAI"

start_time = time.time()
print(f"start time - {start_time}")

pdf_folder_path = "../test_statements/"
documents = []
n_pdfs = 0

api_key = os.getenv("OPENAI_API_KEY")
# azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

if not api_key:
    raise ValueError("API key and Azure endpoint must be set")

# openai_ef = AzureOpenAIEmbeddings(
#     model="text-embedding-ada-002",
#     api_key=SecretStr(api_key),
#     azure_endpoint=azure_endpoint
# )

embeddings = OpenAIEmbeddings(api_key=SecretStr(api_key), model="text-embedding-3-large")

# openai_ef = OllamaEmbeddings(
#     model="llama3.2",
#     base_url="http://austin.local:11434"
# )

for file in os.listdir(pdf_folder_path):
    if file.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder_path, file)
        loader = PyPDFLoader(pdf_path, headers={"path": pdf_path})
        documents.extend(loader.load())
        n_pdfs += 1

print(f"number of pdfs - {n_pdfs}")
print(f"number of documents - {len(documents)}")

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
chunked_documents = text_splitter.split_documents(documents)

client = chromadb.PersistentClient(path="dbstore/")
if COLLECTION_NAME in [ c.name for c in client.list_collections()]:
    print("Collection deleted")
    client.delete_collection(COLLECTION_NAME)

consent_collection = client.create_collection(COLLECTION_NAME)

print(f"number of document chunks - {len(chunked_documents)}")

vector_db = Chroma.from_documents(
    client=client,
    documents=chunked_documents,
    embedding=embeddings,
    # persist_directory="dbstore/"
)

print(f"time - {time.time() - start_time}")
