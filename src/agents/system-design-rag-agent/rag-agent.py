# 1. load data
# 2. index data
# 3. store index
# 4. query

import asyncio
import os
from pathlib import Path

from llama_index.core import Settings, VectorStoreIndex

# -- Load, index, and store data --
# SimpleDirectoryReader only captures text (i.e. text resources)
from llama_index.core import SimpleDirectoryReader
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
    device="cpu"
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
from utils.load_dot_env import load_env_dev
load_env_dev()

from llama_index.llms.anthropic import Anthropic

async def run_query(index: VectorStoreIndex, msg: str):
    llm = Anthropic(
        model="claude-haiku-4-5", 
        max_tokens=1024, 
        temperature=0.1
    )
    
    query_engine = index.as_query_engine(llm=llm)
    
    response = await query_engine.aquery(msg)
    print(response)
    
    
if __name__ == "__main__":
    index = asyncio.run(ingest_docs())
    asyncio.run(run_query(index=index, msg="what are ways I can improve the latency of my db reads when the throughput is around 100K for 1M users"))