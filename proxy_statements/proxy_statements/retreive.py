from langchain_chroma import Chroma
import chromadb

VECTOR_DB_PATH = "chroma_db/"
COLLECTION_NAME = "TEST_COLLECTION"
PDF_FOLDER = "test_statements/"

# ---------------------------------------------------------------------------------------------------------------------

client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

print(f"collections: {client.list_collections()}")

if COLLECTION_NAME in [c.name for c in client.list_collections()]:

    collection = client.get_collection(name=COLLECTION_NAME)

    vector_db = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
    )

    print(vector_db._collection.get(include=["embeddings"])) # type: ignore

else:

    print( f"Collection ({COLLECTION_NAME}) not found." )
