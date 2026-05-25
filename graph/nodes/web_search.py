from typing import Any, Dict
from dotenv import load_dotenv
from langchain_core.documents import Document
#  New, correct import path
from langchain_community.tools import TavilySearchResults

from graph.state import GraphState

load_dotenv()

#  Use TavilySearchResults to get structured list results back
web_search_tool = TavilySearchResults(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]

    # Cleaned up the document state initialization using safe dictionary retrieval
    documents = state.get("documents")
    if documents is None:
        documents = []

    # This invoke now perfectly returns a list of dicts: [{"url": "...", "content": "..."}, ...]
    tavily_results = web_search_tool.invoke({"query": question})

    # Map over the list elements safely using .get() to prevent KeyErrors
    joined_tavily_result = "\n".join(
        [tavily_result.get("content", "") for tavily_result in tavily_results]
    )

    web_results = Document(page_content=joined_tavily_result)
    documents.append(web_results)

    return {"documents": documents, "question": question}


if __name__ == "__main__":
    # Test block works cleanly now!
    res = web_search(state={"question": "agent memory", "documents": None})
    print("\nResult Documents:")
    print(res["documents"][0].page_content)