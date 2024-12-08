from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage

llm = ChatOllama(
    model="llama3.3",
    base_url="http://austin.local:11434",
    format="json",
    verbose=True
)

messages = [
    ("system", "You are a helpful translator. Translate the user sentence to French. Return information in a json format using like the following: {result: ...}"),
    ("human", "I love programming."),
]

ai_msg = llm.invoke(input=messages)

print(ai_msg.content)



