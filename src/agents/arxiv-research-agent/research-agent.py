import asyncio
import json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import arxiv

from llama_index.core import Document
from llama_index.llms.anthropic import Anthropic
from llama_index.core.agent.workflow import FunctionAgent # tool-calling agent from llamaindex
from llama_index.core.workflow import Context
from llama_index.core.workflow import JsonPickleSerializer
from llama_index.core.agent.workflow import AgentStream

ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env.local"
load_dotenv(ENV_PATH)

CTX_FILE_PATH = Path(__file__).resolve().parent / "agent-ctx.json"
    
def save_ctx_to_file(ctx_dict: dict):
    with open(CTX_FILE_PATH, "w") as file:
        json.dump(ctx_dict, file, indent=4)

def load_ctx_from_file():
    try: 
        with open(CTX_FILE_PATH, "r") as file:
            data: dict = json.load(file)
            return data
    except FileNotFoundError:
        return None


def query_arxiv(query: str, sort_by: Optional[str] = "relevance", max_results: Optional[int] = 3):
    """
    Query Arvix for research papers relevant to the user's query.

    Args:
        query (str): The query to be passed to arXiv.
        sort_by (str): Either 'relevance' (default) or 'recent'
    """

    sort = arxiv.SortCriterion.Relevance
    if sort_by == "recent":
        sort = arxiv.SortCriterion.SubmittedDate
        
    search = arxiv.Search(query, max_results=max_results, sort_by=sort)
    search_results = arxiv.Client().results(search=search)
    results = []
    for result in search_results:
        results.append(
            Document(text=f"{result.pdf_url}: {result.title}\n{result.summary}", extra_info={ "links": result.links })
        )
    return results


SYSTEM_PROMPT = """
You are a research agent that finds research papers on Arxiv relevant to the user's query.
"""

if __name__ == "__main__":
    print("Setting env...")
    
    agent = FunctionAgent(
        name="research-agent",
        tools=[query_arxiv],
        llm=Anthropic(model="claude-haiku-4-5", max_tokens=1024),
        system_prompt=SYSTEM_PROMPT,
        streaming=True
    )
    
    previous_ctx = load_ctx_from_file()
    
    print("Loading agent context...")
    
    if previous_ctx:
        ctx = Context.from_dict(agent, data=previous_ctx, serializer=JsonPickleSerializer())
    else:
        ctx = Context(agent)
        
    async def chat():
        print("Agent is thinking...", end="", flush=True)
        
        handler = agent.run(user_msg=query, ctx=ctx, max_iterations=10)
        has_started_responding = False
        
        async for event in handler.stream_events():
            if isinstance(event, AgentStream):
                if (has_started_responding is not True):
                    print("\r\033[KAgent: ", end="", flush=True)
                    has_started_responding = True
                    
                print(event.delta, end="", flush=True)
            
        save_ctx_to_file(ctx_dict=ctx.to_dict(JsonPickleSerializer()))
    
    print("Agent is ready!")
    print("Hi, what can I help you with research today?")
    while True:
        query = input("You (enter q to exit): ")
        if query.lower() == "q":
            print("Ok, goodbye!")
            break
        
        asyncio.run(chat())
        print()
    
