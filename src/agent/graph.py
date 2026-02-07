"""LangGraph workflow definition for the FOMC RAG agent."""

from functools import partial

from langgraph.graph import StateGraph, END

from src.agent.nodes import (
    assess_query,
    evaluate_confidence,
    reformulate_query,
    respond,
    search_corpus,
    synthesize_answer,
    validate_citations,
)
from src.agent.state import AgentState


def _route_after_assess(state: AgentState) -> str:
    """Route based on whether retrieval is needed."""
    if state.get("needs_retrieval"):
        return "search_corpus"
    return "respond"


def _route_after_confidence(state: AgentState) -> str:
    """Route based on confidence level per agent-workflow.md."""
    confidence = state.get("confidence", "insufficient")
    attempts = state.get("reformulation_attempts", 0)

    if confidence in ("high", "medium"):
        return "synthesize_answer"
    elif confidence == "low" and attempts < 2:
        return "reformulate_query"
    else:
        # insufficient or max retries reached
        return "respond"


def build_graph(search_fn) -> StateGraph:
    """Build the LangGraph agent workflow.

    Args:
        search_fn: A callable (query: str, top_k: int) -> list[ChunkResult]
                   that searches the FOMC corpus.

    Returns:
        A compiled LangGraph StateGraph.
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("assess_query", assess_query)
    graph.add_node("search_corpus", partial(search_corpus, search_fn=search_fn))
    graph.add_node("evaluate_confidence", evaluate_confidence)
    graph.add_node("reformulate_query", reformulate_query)
    graph.add_node("synthesize_answer", synthesize_answer)
    graph.add_node("validate_citations", validate_citations)
    graph.add_node("respond", respond)

    # Set entry point
    graph.set_entry_point("assess_query")

    # Add edges
    graph.add_conditional_edges("assess_query", _route_after_assess)
    graph.add_edge("search_corpus", "evaluate_confidence")
    graph.add_conditional_edges("evaluate_confidence", _route_after_confidence)
    graph.add_edge("reformulate_query", "search_corpus")
    graph.add_edge("synthesize_answer", "validate_citations")
    graph.add_edge("validate_citations", "respond")
    graph.add_edge("respond", END)

    return graph.compile()
