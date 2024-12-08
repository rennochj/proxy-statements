import os
import chromadb
from uuid import uuid4
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from pydantic import SecretStr

# ----

COLLECTION_NAME = "collection_name"

# ----

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API key and Azure endpoint must be set")

embeddings = OpenAIEmbeddings(api_key=SecretStr(secret_value=api_key), model="text-embedding-3-large")

# ----

from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db",  # Where to save data locally, remove if not necessary
)

# ----

persistent_client = chromadb.PersistentClient()
persistent_client.delete_collection(name=COLLECTION_NAME)
collection = persistent_client.get_or_create_collection(name=COLLECTION_NAME)

vector_store_from_client = Chroma(
    client=persistent_client,
    collection_name="collection_name",
    embedding_function=embeddings,
)

# ----

document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
    id=1,
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 80 degrees.",
    metadata={"source": "news"},
    id=2,
)

document_20 = Document(
    page_content="The weather forecast for Friday is cloudy and overcast, with a high of 90 degrees.",
    metadata={"source": "news"},
    id=20,
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
    id=3,
)

document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"},
    id=4,
)

document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "tweet"},
    id=5,
)

document_6 = Document(
    page_content="Is the new iPhone worth the price? Read this review to find out.",
    metadata={"source": "website"},
    id=6,
)

document_7 = Document(
    page_content="The top 10 soccer players in the world right now.",
    metadata={"source": "website"},
    id=7,
)

document_8 = Document(
    page_content="LangGraph is the best framework for building stateful, agentic applications!",
    metadata={"source": "tweet"},
    id=8,
)

document_9 = Document(
    page_content="The stock market is down 500 points today due to fears of a recession.",
    metadata={"source": "news"},
    id=9,
)

document_10 = Document(
    page_content="I have a bad feeling I am going to get deleted :(",
    metadata={"source": "tweet"},
    id=10,
)

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
    document_20,
]

uuids = [str(uuid4()) for _ in range(len(documents))]

vector_store.add_documents(documents=documents, ids=uuids)

# ----

# updated_document_1 = Document(
#     page_content="I had chocolate chip pancakes and fried eggs for breakfast this morning.",
#     metadata={"source": "tweet"},
#     id=1,
# )

# updated_document_2 = Document(
#     page_content="The weather forecast for tomorrow is sunny and warm, with a high of 82 degrees.",
#     metadata={"source": "news"},
#     id=2,
# )

# vector_store.update_document(document_id=uuids[0], document=updated_document_1)
# # You can also update multiple documents at once
# vector_store.update_documents(
#     ids=uuids[:2], documents=[updated_document_1, updated_document_2]
# )

# ----

# vector_store.delete(ids=uuids[-1])

# ----

results = vector_store.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy",
    k=2,
    filter={"source": "tweet"},
)
for res in results:
    print(f"* {res.page_content} [{res}]")

# ----

results = vector_store.similarity_search_with_score(
    "Will it be hot tomorrow or Friday?", k=2, filter={"source": "news"}, 
)
for res, score in results:
    print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

# ----

results = vector_store.similarity_search_by_vector(
    embedding=embeddings.embed_query("I love green eggs and ham!"), k=1
)
for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")

# ----

retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 2, "fetch_k": 5}
)

result = retriever.invoke("Stealing from the bank is a crime", filter={"source": "news"})
for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")

print(vector_store._collection.get(include=['embeddings'])) # type: ignore
