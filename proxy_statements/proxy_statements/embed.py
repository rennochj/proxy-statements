import os
from uuid import uuid4

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb
from pydantic import SecretStr

VECTOR_DB_PATH = "chroma_db"
COLLECTION_NAME = "TEST_COLLECTION"
PDF_FOLDER = "test_statements/"

documents = []
n_pdfs = 0

# ---------------------------------------------------------------------------------------------------------------------

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API key and Azure endpoint must be set")

# ---------------------------------------------------------------------------------------------------------------------

embeddings = OpenAIEmbeddings(api_key=SecretStr(api_key), model="text-embedding-3-large")
client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

if COLLECTION_NAME in [ c.name for c in client.list_collections() ]:
    client.delete_collection(name=COLLECTION_NAME)

collection = client.create_collection(name=COLLECTION_NAME)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)

# ---------------------------------------------------------------------------------------------------------------------

for file in os.listdir(PDF_FOLDER):
    if file.endswith('.pdf'):
        file_path = os.path.join(PDF_FOLDER, file)
        loader = PyPDFLoader(file_path, headers={"path": file_path})
        documents.extend(loader.load())
        n_pdfs += 1

# vector_db = Chroma(
#     client=client,
#     embedding_function=embeddings,
#     persist_directory=VECTOR_DB_PATH,
#     collection_name=COLLECTION_NAME
# )

# vector_db.add_documents(
#     documents=text_splitter.split_documents(documents),
#     ids=[str(uuid4()) for _ in range(len(documents))]
# )

vector_db = Chroma.from_documents(
    client=client,
    embedding=embeddings,
    persist_directory=VECTOR_DB_PATH,
    collection_name=COLLECTION_NAME,
    documents=text_splitter.split_documents(documents),
    ids=[str(uuid4()) for _ in range(len(documents))]
)

# ---------------------------------------------------------------------------------------------------------------------

print(vector_db._collection.get(include=['embeddings'])) # type: ignore


