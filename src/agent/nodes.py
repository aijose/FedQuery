"""Agent graph nodes for the FOMC RAG workflow."""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.llm.config import get_llm

logger = logging.getLogger(__name__)

# Confidence thresholds calibrated for all-MiniLM-L6-v2 cosine similarity
# on the FOMC corpus (relevant queries typically score 0.55-0.70)
CONFIDENCE_THRESHOLDS = {
    "high": 0.55,
    "medium": 0.40,
    "low": 0.25,
}


def evaluate_confidence_level(avg_score: float) -> str:
    """Map average relevance score to confidence level.

    high ≥ 0.55, medium ≥ 0.40, low ≥ 0.25, insufficient < 0.25.
    """
    if avg_score >= CONFIDENCE_THRESHOLDS["high"]:
        return "high"
    elif avg_score >= CONFIDENCE_THRESHOLDS["medium"]:
        return "medium"
    elif avg_score >= CONFIDENCE_THRESHOLDS["low"]:
        return "low"
    else:
        return "insufficient"


def assess_query(state: AgentState) -> dict:
    """Analyze query and determine if retrieval is needed.

    Uses LLM to classify query. Most FOMC-related queries need retrieval.
    """
    llm = get_llm()
    messages = [
        SystemMessage(content=(
            "You are a query classifier for an FOMC document retrieval system. "
            "Determine if the user's question requires searching FOMC documents. "
            "Respond with ONLY 'yes' or 'no'.\n"
            "- Questions about Fed policy, interest rates, inflation, economic outlook, "
            "FOMC meetings → 'yes'\n"
            "- Greetings, general knowledge, non-FOMC topics → 'no'"
        )),
        HumanMessage(content=state["query"]),
    ]
    response = llm.invoke(messages)
    needs_retrieval = "yes" in response.content.lower()
    return {"needs_retrieval": needs_retrieval}


def search_corpus(state: AgentState, search_fn) -> dict:
    """Search the FOMC corpus using the provided search function.

    search_fn should accept (query: str, top_k: int) and return
    a list of ChunkResult dicts.
    """
    query = state.get("reformulated_query") or state["query"]
    results = search_fn(query, top_k=10)
    return {"retrieved_chunks": results}


def evaluate_confidence(state: AgentState) -> dict:
    """Assess relevance of retrieved chunks and set confidence level."""
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {"confidence": "insufficient"}

    avg_score = sum(c["relevance_score"] for c in chunks) / len(chunks)
    confidence = evaluate_confidence_level(avg_score)
    return {"confidence": confidence}


def reformulate_query(state: AgentState) -> dict:
    """Rephrase the query for better retrieval.

    Uses the LLM to reformulate based on original query and what was retrieved.
    """
    llm = get_llm()
    chunks_summary = "\n".join(
        f"- {c['chunk_text'][:100]}..." for c in state.get("retrieved_chunks", [])[:3]
    )

    messages = [
        SystemMessage(content=(
            "You are a query reformulation expert for FOMC document search. "
            "The original query did not retrieve sufficiently relevant results. "
            "Rephrase the query to improve retrieval. Keep it concise and focused "
            "on FOMC-specific terminology. Respond with ONLY the reformulated query."
        )),
        HumanMessage(content=(
            f"Original query: {state['query']}\n"
            f"Retrieved passages (low relevance):\n{chunks_summary}\n"
            f"Reformulated query:"
        )),
    ]
    response = llm.invoke(messages)
    attempts = state.get("reformulation_attempts", 0) + 1
    return {
        "reformulated_query": response.content.strip(),
        "reformulation_attempts": attempts,
    }


def synthesize_answer(state: AgentState) -> dict:
    """Generate an answer grounded in retrieved chunks with citations."""
    llm = get_llm()
    chunks = state.get("retrieved_chunks", [])

    context = "\n\n".join(
        f"[Source {i+1}] {c['document_name']} ({c['document_date']}), "
        f"§{c['section_header']}\n{c['chunk_text']}"
        for i, c in enumerate(chunks)
    )

    messages = [
        SystemMessage(content=(
            "You are a research assistant answering questions about FOMC monetary policy. "
            "Answer ONLY based on the provided source passages. "
            "For each claim, cite the source using [Source N] notation. "
            "If the sources don't contain enough information, say so. "
            "Be precise and factual."
        )),
        HumanMessage(content=(
            f"Question: {state['query']}\n\n"
            f"Source passages:\n{context}\n\n"
            "Answer:"
        )),
    ]
    response = llm.invoke(messages)

    # Build citations from the chunks used
    citations = []
    for i, chunk in enumerate(chunks):
        citations.append({
            "document_name": chunk["document_name"],
            "document_date": chunk["document_date"],
            "section_header": chunk["section_header"],
            "chunk_id": chunk["chunk_id"],
            "relevance_score": chunk["relevance_score"],
            "quoted_excerpt": chunk["chunk_text"][:200],
        })

    return {"answer": response.content, "citations": citations}


def validate_citations(state: AgentState) -> dict:
    """Verify each citation maps to an actual retrieved chunk."""
    valid_chunk_ids = {c["chunk_id"] for c in state.get("retrieved_chunks", [])}
    chunk_texts = {c["chunk_id"]: c["chunk_text"] for c in state.get("retrieved_chunks", [])}

    validated = []
    for citation in state.get("citations", []):
        if citation["chunk_id"] in valid_chunk_ids:
            # Verify quoted excerpt is from the chunk text
            chunk_text = chunk_texts.get(citation["chunk_id"], "")
            excerpt = citation.get("quoted_excerpt", "")
            if excerpt and excerpt in chunk_text:
                validated.append(citation)
            else:
                # Keep citation but with truncated excerpt that exists
                citation["quoted_excerpt"] = chunk_text[:200] if chunk_text else ""
                validated.append(citation)

    return {"citations": validated}


def respond(state: AgentState) -> dict:
    """Format the final response — grounded answer or uncertainty message."""
    confidence = state.get("confidence", "insufficient")
    answer = state.get("answer")
    citations = state.get("citations", [])

    if confidence == "insufficient" or not answer:
        # Uncertainty response
        queries_searched = [state["query"]]
        if state.get("reformulated_query"):
            queries_searched.append(state["reformulated_query"])

        chunks = state.get("retrieved_chunks", [])
        best_matches = "\n".join(
            f"  [{i+1}] {c['document_name']} (score: {c['relevance_score']:.2f})"
            for i, c in enumerate(chunks[:3])
        )

        answer = (
            "I was unable to find sufficient information in the FOMC document corpus "
            "to answer this question confidently. The available documents may not "
            "cover this topic.\n\n"
            f"Searched: {', '.join(queries_searched)}\n"
            f"Best matches (low relevance):\n{best_matches}"
        )
        return {"answer": answer, "citations": []}

    # Grounded answer with sources
    sources = "\n".join(
        f"  [{i+1}] {c['document_name']}, {c['document_date']}, "
        f"§{c['section_header']} (chunk {c['chunk_id'][:8]})"
        for i, c in enumerate(citations)
    )
    formatted = f"{answer}\n\nSources:\n{sources}" if sources else answer
    return {"answer": formatted}
