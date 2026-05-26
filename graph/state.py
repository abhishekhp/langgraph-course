from typing import List, TypedDict, Annotated
from langchain_core.documents import Document


# Define the explicit, type-safe reducer function
def reduce_documents(left: List[Document], right: List[Document]) -> List[Document]:
    """Combines document lists from parallel or sequential nodes safely."""
    left = left or []
    right = right or []
    return left + right


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents (Managed safely by a custom reducer)
    """

    question: str
    generation: str
    web_search: bool

    #  New bulletproof definition:
    documents: Annotated[List[Document], reduce_documents]