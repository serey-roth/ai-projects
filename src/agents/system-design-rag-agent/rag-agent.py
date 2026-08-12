# 1. load data
# 2. index data
# 3. store index
# 4. query

import asyncio
import os
from pathlib import Path

from utils.load_dot_env import load_env_dev
load_env_dev()

# -- Set up tracing and observability --
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register

tracer_provider = register(project_name="system-design-rag-agent", protocol="http/protobuf")
LlamaIndexInstrumentor().instrument(
    tracer_provider=tracer_provider
)

# -- Load, index, and store data --
# SimpleDirectoryReader only captures text (i.e. text resources)
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.readers.file import PDFReader
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.kvstore import SimpleKVStore

import chromadb

DATA_DIR = Path(__file__).resolve().parent / "data"

CHROMADB_PATH = Path(__file__).resolve().parent / "chroma-db"

KV_STORE_PATH = Path(__file__).resolve().parent / "kv-store.json"

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device="cpu",
)

async def ingest_docs():
    reader = SimpleDirectoryReader(
        input_dir=DATA_DIR,
        required_exts=[".pdf"], 
        file_extractor={ ".pdf": PDFReader() },
        filename_as_id=True,
        exclude_empty=True,
        exclude_hidden=True,
        
    )
    
    documents = await reader.aload_data(show_progress=False)
    
    db = chromadb.PersistentClient(path=str(CHROMADB_PATH))
    chroma_collection = db.get_or_create_collection("system-design")
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    if os.path.exists(KV_STORE_PATH):
        kv_store = SimpleKVStore.from_persist_path(KV_STORE_PATH)
    else:
        kv_store = SimpleKVStore()
        
    doc_store = SimpleDocumentStore(simple_kvstore=kv_store, namespace="system-design-doc-store")
    
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=1024, chunk_overlap=20),
            HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5",
                device="cpu"
            )
        ],
        vector_store=vector_store,
        docstore=doc_store,
        docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE
    )
        
    await pipeline.arun(documents=documents, num_workers=4)
    
    kv_store.persist(persist_path=KV_STORE_PATH)
    
    index = VectorStoreIndex.from_vector_store(vector_store)
    return index

# -- Query --

from llama_index.llms.anthropic import Anthropic
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine.types import BaseChatEngine

memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

def create_chat_engine(index: VectorStoreIndex):
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
    
    return chat_engine 
    
if __name__ == "__main__":
    print("Ingesting docs...")
    index = asyncio.run(ingest_docs())
    
    print("Loading chat engine...")
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
                
    print("Agent is ready!")
    print("Hi, what system design question do you have today?")
    while True:
        query = input("You (enter q to exit): ")
        if query.lower() == "q":
            print("Ok, goodbye!")
            break
        
        asyncio.run(ask_question(chat_engine=chat_engine, message=query))
        print()