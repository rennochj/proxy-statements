import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain_chroma import Chroma
import chromadb
from pydantic import SecretStr

# ---------------------------------------------------------------------------------------------------------------------

VECTOR_DB_PATH = "chroma_db/"
COLLECTION_NAME = "TEST_COLLECTION"
PDF_FOLDER = "test_statements/"

# ---------------------------------------------------------------------------------------------------------------------

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API key and Azure endpoint must be set")

embeddings = OpenAIEmbeddings(api_key=SecretStr(api_key), model="text-embedding-3-large")

model = ChatOpenAI(
    api_key=SecretStr(api_key),
    model="gpt-4"
)

# ---------------------------------------------------------------------------------------------------------------------

client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

vector_db = Chroma(
    client=client,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
)

# ---------------------------------------------------------------------------------------------------------------------

parser = StrOutputParser()

# ---------------------------------------------------------------------------------------------------------------------

system_template = "Answer the user's questions based on the below context:\\n\\n{context}"
user_template = "{input}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", user_template)]
)

# ---------------------------------------------------------------------------------------------------------------------

retriever = vector_db.as_retriever( )

combine_docs_chain = create_stuff_documents_chain(
    model, prompt_template
)

retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# ---------------------------------------------------------------------------------------------------------------------

chain = retrieval_chain | prompt_template | model | parser

input = {
    "input": "Summarize the compensation data for the CEO of Abbot."
}

result = chain.invoke(input=input )
print( result )