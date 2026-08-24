import os
import asyncio
from deepeval import evaluate

from utils.load_dot_env import load_env_dev
load_env_dev()

os.environ["CONFIDENT_API_KEY"] = os.getenv("CONFIDENT_API_KEY")
os.environ["ANTHROPIC_AI_KEY"] = os.getenv("ANTHROPIC_API_KEY")
os.environ["CONFIDENT_TRACE_VERBOSE"] = "0"

from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.llms.anthropic import Anthropic
from llama_index.core.base.base_query_engine import BaseQueryEngine
import llama_index.core.instrumentation as instrument
from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

from deepeval.integrations.llama_index import (
    instrument_llama_index,
)
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import LLMTestCase, RetrievedContextData
from deepeval.metrics import AnswerRelevancyMetric, BaseMetric, ContextualRelevancyMetric
from deepeval.models import AnthropicModel, OllamaModel

from ingestion import ingest_docs, load_index

instrument_llama_index(instrument.get_dispatcher())

TEST_QUERIES = [
    "How do I speed up my read performance for an app that has 1M daily active users?",
    "If I add read replicas to scale reads, how do I keep them in sync after a write?",
    "How do I improve my search latency for complex queries?",
    "For a news feed, how I do handle posts of a popular user who has hundred of thousands of users?",
]

MODEL_TYPE = "anthropic" # or ollama
SIMILARITY_TOP_K = 2

# - Contexts contain a lot of irrelevant info, including raw tables that aren't related to the query. 
# - For raw tables, we can a step during ingestion to classify and only summarize meaningful tables, 
#   though making sure summaries are meaningful is another task. Also summary drift is a concern as well.
# - Adding a reranker as a node postprocessor to filter useless chunks could help with the noise.

def create_query_engine(index: VectorStoreIndex):
    print("Initializing query engine (Llama3:8b)...")
    if MODEL_TYPE == "ollama":
        llm = Ollama(
            model="llama3:8b", 
            base_url="http://localhost:11434",
            temperature=0.0,
        )
    else:
        llm = Anthropic(
            model="claude-haiku-4-5", 
            temperature=0.0
        )

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=SIMILARITY_TOP_K,
    )

    response_synthesizer = get_response_synthesizer(
        llm=llm
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.5)
        ],
    )

    return query_engine


def create_eval_dataset():
    goldens = [Golden(input=q) for q in TEST_QUERIES]
    dataset = EvaluationDataset(goldens=goldens)
    return dataset
    
    
def create_eval_metrics():
    if MODEL_TYPE == "ollama":
        llm = OllamaModel(
            model="llama3:8b", 
            base_url="http://localhost:11434",
            temperature=0.0,
        )
    else:
        llm = AnthropicModel(
            model="claude-haiku-4-5", 
            temperature=0.0
        )
    
    return [
        AnswerRelevancyMetric(model=llm),
        ContextualRelevancyMetric(model=llm)
    ]


def run_evals(query_engine: BaseQueryEngine, dataset: EvaluationDataset, metrics: list[BaseMetric]):
    for golden in dataset.goldens:
        response = query_engine.query(golden.input)
        response_text = response.response
        retrieval_context=[RetrievedContextData(
            source=n.node.metadata.get("file_name", ""),
            context=n.node.get_content()
        ) for n in response.source_nodes]
        
        test_case = LLMTestCase(
            input=golden.input,
            actual_output=response_text,
            retrieval_context=retrieval_context
        )
        
        dataset.add_test_case(test_case)
    
    result = evaluate(
        test_cases=dataset.test_cases,
        metrics=metrics,
        hyperparameters={
            "model": "anthropic/claude-haiku-4-5" if MODEL_TYPE == "anthropic" else "ollama/llama3:8b",
            "similarity_top_k": SIMILARITY_TOP_K
        }
    )
    return result.test_results
    

if __name__ == "__main__":
    print("Preparing docs for agent...")
    index = load_index()
    if index is None:
        print("Ingesting docs...")
        index = asyncio.run(ingest_docs())
    else:
        print("Docs have been ingested.")
    print("Docs ready!")

    print("Preparing query engine...")
    query_engine = create_query_engine(index)
    print("Query engine ready!")
    
    print("Preparing evaluator...")
    dataset = create_eval_dataset()
    metrics = create_eval_metrics()
    print("Evaluator ready!")
    
    print("Evaluating test queries...")
    results = run_evals(
        query_engine=query_engine,
        dataset=dataset,
        metrics=metrics
    )
    
    print("Evaluation finished! ")