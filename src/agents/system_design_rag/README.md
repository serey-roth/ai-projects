# Project Description

A RAG Q&A for System Design Interview with LlamaIndex.

System design interviews are difficult, even more so when there's a lot of material to go through. This is built to aid that process by answering specific user
queries grounded in the prep material, via a simple RAG pipeline.

This is an ongoing project so things are still moving around, but as of now, you can ask questions regarding the PDF files in the `/data` folder. 
There are 4 said files, which are PDF exports of "Hello Interview" web pages. I was a member of Hello Interview at the time I was exporting these documents.

## Data Ingestion

Each PDF is converted to a Markdown file, via `docling`. Each Markdown file is then cleaned up to preserve atomic elements such as code blocks, images, image captions, tables and lists.

`docling` has options to enrich images, tables and code blocks, but enabling any of them add huge latency to the markdown conversion since `docling` needs to pull down the appropriate transformer.
Instead, we handle said enrichment manually by integrating a VLM enrichment step into the conversion. As of now, we support image enrichment with image extraction. 

If `do-picture-enrichment=True` for `convert_pdf_to_markdown`, then the Markdown file will contain descriptions of the images present in PDF file as well as a link to the extracted image in the `/data/images/[file_name]/[image_hash]`.
As of now, the image is hashed against the actual cropped content.

Once the PDF is converted, it gets parsed into nodes via `MarkdownNodeParser` that captures the markdown elements and then a block-aware version of `SemanticDoubleMergingNodeParserSplitter` that preservers block spacing and atomic elements mentioned above.

The nodes are then indexed into a local `ChromaDB` vector store, with document tracking via `llamaindex`'s `SimpleDocumentStore` with `DocstoreStrategy.UPSERTS_AND_DELETE` to prevent duplicates. The current embedding model is `BAAI/bge-small-en-v1.5` from HuggingFace.

## Query and Retrieval

As of now, we're using `llamaindex`'s pre-built chat engine with `condense_plus_context` mode. That said, we might configure the engine ourselves as early evals have indicated poor context retrieval. With `top-k=2` and similarity threshold of 0.5, the retrieved context contains
a lot of noise and irrelevant info. That is our current task, to improve context through evaluations. 

Evaluations is currently done through `deepeval` with 2 metrics: `AnswerRelevancy` and `ContextualRelevancy`. 

