
import asyncio
import os
from pathlib import Path

from utils.load_dot_env import load_env_dev
load_env_dev()

# SimpleDirectoryReader only captures text (i.e. text resources)
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext
from llama_index.readers.file import MarkdownReader
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import MarkdownNodeParser, SemanticDoubleMergingSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.kvstore import SimpleKVStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core import (
    load_index_from_storage,
)

import chromadb

DATA_DIR = Path(__file__).resolve().parent / "data"

CHROMADB_PATH = Path(__file__).resolve().parent / "chroma_db"

STORAGE_CONTEXT_DIR = Path(__file__).resolve().parent / "storage_context"

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device="cpu",
)

from pdf_to_markdown import convert_pdf_to_markdown
from sematic_splitter_node_parser import block_aware_sentence_splitter, BlockAwareSemanticDoubleMergingSplitterNodeParser

def _save_markdown_to_file(output_file_path: Path, markdown: str):
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(markdown)
        

def convert_pdfs_to_markdowns():
    with os.scandir(DATA_DIR) as entries:
        pdf_paths = [Path(e.path) for e in entries if e.is_file() and e.name.endswith(".pdf")]

    print(f"Found {len(pdf_paths)} PDF(s) in {DATA_DIR}")

    for i, pdf_path in enumerate(pdf_paths, start=1):
        print(f"[{i}/{len(pdf_paths)}] Parsing {pdf_path.name}...")
        markdown = convert_pdf_to_markdown(pdf_path, do_picture_enrichment=True)

        output_file_path = DATA_DIR / f"{pdf_path.stem}.md"
        _save_markdown_to_file(
            output_file_path=output_file_path,
            markdown=markdown
        )
        print(f"[{i}/{len(pdf_paths)}] Saved {output_file_path.name}")

    print("All PDFs converted.")


def get_vector_store():
    print("Connecting to Chroma vector store...")
    db = chromadb.PersistentClient(path=str(CHROMADB_PATH))
    chroma_collection = db.get_or_create_collection("system_design_rag")
    return ChromaVectorStore(chroma_collection=chroma_collection)
    
    
def get_persisted_storage_context():
    if not (os.path.exists(STORAGE_CONTEXT_DIR)):
        print("Storage context not found!")
        return None
    
    print("Loading persisted storage context...")
    doc_store = SimpleDocumentStore.from_persist_dir(persist_dir=STORAGE_CONTEXT_DIR, namespace="system_design_rag")
    index_store = SimpleIndexStore.from_persist_dir(persist_dir=STORAGE_CONTEXT_DIR)
    vector_store = get_vector_store()
    
    return StorageContext.from_defaults(
        docstore=doc_store,
        index_store=index_store,
        vector_store=vector_store,
        persist_dir=STORAGE_CONTEXT_DIR
    )
    

def create_new_storage_context():
    print("Creating storage context...")
    doc_store = SimpleDocumentStore(simple_kvstore=SimpleKVStore(), namespace="system_design_rag")
    index_store = SimpleIndexStore(simple_kvstore=SimpleKVStore())
    vector_store = get_vector_store()
    
    return StorageContext.from_defaults(
        docstore=doc_store,
        index_store=index_store,
        vector_store=vector_store,
        persist_dir=STORAGE_CONTEXT_DIR
    )


async def ingest_docs():
    print(f"Converting PDF documents from {DATA_DIR}...")
    convert_pdfs_to_markdowns()

    print(f"Loading markdown documents from {DATA_DIR}...")
    reader = SimpleDirectoryReader(
        input_dir=DATA_DIR,
        required_exts=[".md"],
        filename_as_id=True,
        exclude_empty=True,
        exclude_hidden=True,
    )

    documents = await reader.aload_data(show_progress=False)
    print(f"Loaded {len(documents)} document(s)")

    storage_context = get_persisted_storage_context() or create_new_storage_context()

    embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device="cpu")
    node_parser = MarkdownNodeParser(include_metadata=True, include_prev_next_rel=True)
    splitter = BlockAwareSemanticDoubleMergingSplitterNodeParser(
        include_metadata=True,
        include_prev_next_rel=True,
        embed_model=HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            device="cpu",
        ),
        sentence_splitter=block_aware_sentence_splitter,
        max_chunk_size=1600,
        initial_threshold=0.5,
        appending_threshold=0.6,
        merging_threshold=0.6
    )
    
    pipeline = IngestionPipeline(
        transformations=[
            splitter,
            node_parser,
            embedding_model
        ],
        vector_store=storage_context.vector_store,
        docstore=storage_context.docstore,
        docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE,
    )

    print("Running ingestion pipeline (chunking + embedding)...")
    nodes = await pipeline.arun(documents=documents, num_workers=4)
    print(f"Ingestion complete: {len(nodes)} node(s) upserted")

    index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
    print("Vector index ready.")
    
    storage_context.persist(persist_dir=STORAGE_CONTEXT_DIR)
    
    return index

    
def load_index():
    if not os.path.exists(STORAGE_CONTEXT_DIR):
        return None
    
    storage_context = get_persisted_storage_context()
    return load_index_from_storage(storage_context=storage_context)
