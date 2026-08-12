# 1. load data
# 2. index data
# 3. store index
# 4. query

from pathlib import Path
from typing import Sequence

from llama_index.core import Document, Settings, VectorStoreIndex

# -- Load data --
# SimpleDirectoryReader only captures text (i.e. text resources)
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PDFReader

DATA_DIR = Path(__file__).resolve().parent / "data"
    
def load_data():
    reader = SimpleDirectoryReader(
        input_dir=DATA_DIR,
        required_exts=[".pdf"], 
        file_extractor={ ".pdf": PDFReader() }
    )
    documents = reader.load_data(show_progress=False)
    return documents

# -- Index and Store 
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

CHROMADB_PATH = Path(__file__).resolve().parent / "chroma_db"

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

def index_documents_and_store(docs: Sequence[Document]):
    db = chromadb.PersistentClient(path=str(CHROMADB_PATH))
    chroma_collection = db.get_or_create_collection("system-design")
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_documents(
        documents=docs, 
        storage_context=storage_context,
    ) # might want to add transformations
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
    documents = load_data()
    index = index_documents_and_store(docs=documents)
    # asyncio.run(run_query(index=index, msg="what are ways I can improve the latency of my db reads when the throughput is around 100K for 1M users"))
    print(index.ref_doc_info)