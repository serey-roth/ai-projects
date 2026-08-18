import asyncio

from utils.load_dot_env import load_env_dev
load_env_dev()

# -- Set up tracing and observability --
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register

tracer_provider = register(project_name="system-design-rag-agent", protocol="http/protobuf")
LlamaIndexInstrumentor().instrument(
    tracer_provider=tracer_provider
)
    
from llama_index.core.chat_engine.types import BaseChatEngine
from ingestion import ingest_docs, load_index
from llama_index.core import VectorStoreIndex
from llama_index.llms.anthropic import Anthropic
from llama_index.core.memory import ChatMemoryBuffer

memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

def create_chat_engine(index: VectorStoreIndex):
    print("Initializing chat engine (Claude Haiku)...")
    llm = Anthropic(
            model="claude-haiku-4-5", 
            max_tokens=1024, 
            temperature=0.1
        )
        
    chat_engine = index.as_chat_engine(
        llm=llm,
        chat_mode="condense_plus_context",
        memory=memory,
        context_prompt=(
            """
            You are an expert tutor in system design for technical software engineering interviews. Your goal is to help the current user
            learn and improve their knowledge in system design."
            "Here are the relevant documents for the context:\n"
            "{context_str}"
            "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
            """
        ),
        verbose=False,
    )

    print("Chat engine ready.")
    return chat_engine
    
if __name__ == "__main__":
    print("Preparing docs for agent...")
    index = load_index()
    if index is None:
        print("Ingesting docs...")
        index = asyncio.run(ingest_docs())
    else:
        print("Docs have been ingested.")
    print("Docs ready!")

    chat_engine = create_chat_engine(index)

    async def ask_question(chat_engine: BaseChatEngine, message: str):
        print("Agent is thinking...", end="", flush=True)
        
        response = await chat_engine.astream_chat(message)
        has_started_responding = False
        
        async for token in response.async_response_gen():
            if (has_started_responding is not True):
                print("\r\033[KAgent: ", end="", flush=True)
                has_started_responding = True
                
            print(token, end="", flush=True)
                
    print("Hi, what system design question do you have today?")
    while True:
        query = input("You (enter q to exit): ")
        if query.lower() == "q":
            print("Ok, goodbye!")
            break
        
        asyncio.run(ask_question(chat_engine=chat_engine, message=query))
        print()
