from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]

    documents = retriever.invoke(question)

    # return {"documents": documents, "question": question}

    #  New, correct line: Only return the retrieved documents list
    return {"documents": documents}
