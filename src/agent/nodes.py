"""Agent graph nodes for the FOMC RAG workflow."""

import json
import logging
import re

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
    """Analyze query and determine if retrieval is needed, with date hints.

    Uses LLM to classify query, extract temporal signals, and estimate
    how many results are needed (top_k_hint) based on query scope.
    """
    llm = get_llm()
    messages = [
        SystemMessage(content=(
            "You are a query classifier for an FOMC document retrieval system. "
            "Determine if the user's question requires searching FOMC documents, "
            "extract any temporal date range, and estimate how many search results "
            "are needed.\n\n"
            "Respond with ONLY a JSON object (no markdown, no explanation):\n"
            '{"needs_retrieval": true/false, "date_start": "YYYY-MM-DD" or null, '
            '"date_end": "YYYY-MM-DD" or null, "top_k_hint": integer or null}\n\n'
            "Rules:\n"
            "- Questions about Fed policy, interest rates, inflation, economic outlook, "
            "FOMC meetings → needs_retrieval: true\n"
            "- Greetings, general knowledge, non-FOMC topics → needs_retrieval: false\n"
            "- If the query mentions a specific month/year (e.g. 'December 2024'), "
            "set date_start to the 1st and date_end to the last day of that month\n"
            "- If a year is mentioned without a month (e.g. '2024'), use the full year range\n"
            "- If no temporal signal, set both dates to null\n"
            "- top_k_hint: estimate how many search results are needed to fully answer "
            "the question. The FOMC meets ~8 times per year and documents use very "
            "similar language across meetings, so retrieval needs extra results to "
            "cover all relevant dates. Guidelines: single meeting → null (default 10), "
            "2-3 meetings → 15, full year (~8 meetings) → 30, multi-year → 40-50. "
            "For narrow questions about one topic at one meeting, use null."
        )),
        HumanMessage(content=state["query"]),
    ]
    response = llm.invoke(messages)

    # Parse JSON response; fall back to yes/no heuristic if parsing fails
    needs_retrieval = False
    metadata_hints = None
    top_k_hint = None

    # Strip markdown code fences if present (LLMs often wrap JSON in ```json ... ```)
    text = response.content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        needs_retrieval = bool(parsed.get("needs_retrieval", False))
        date_start = parsed.get("date_start")
        date_end = parsed.get("date_end")
        if date_start and date_end:
            metadata_hints = {"date_start": date_start, "date_end": date_end}
        raw_hint = parsed.get("top_k_hint")
        if isinstance(raw_hint, int) and 1 <= raw_hint <= 50:
            top_k_hint = raw_hint
    except (json.JSONDecodeError, AttributeError):
        logger.debug("assess_query JSON parse failed, falling back to yes/no heuristic")
        needs_retrieval = "yes" in response.content.lower() or "true" in response.content.lower()

    return {
        "needs_retrieval": needs_retrieval,
        "metadata_hints": metadata_hints,
        "top_k_hint": top_k_hint,
    }


def search_corpus(state: AgentState, search_fn) -> dict:
    """Search the FOMC corpus using the provided search function.

    Uses two-pass retrieval when date hints are present:
    1. Filtered pass with date range where clause (priority results)
    2. Unfiltered pass to fill remaining slots
    Results are merged and deduped to top_k (from top_k_hint or default 10).

    search_fn should accept (query: str, top_k: int, where: dict | None)
    and return a list of ChunkResult dicts.
    """
    query = state.get("reformulated_query") or state["query"]
    hints = state.get("metadata_hints")
    top_k = state.get("top_k_hint") or 10

    if hints and hints.get("date_start"):
        where = {"$and": [
            {"document_date": {"$gte": hints["date_start"]}},
            {"document_date": {"$lte": hints["date_end"]}},
        ]}
        filtered = search_fn(query, top_k=top_k, where=where)
        unfiltered = search_fn(query, top_k=top_k)

        # Merge: filtered results first (priority), then fill from unfiltered
        seen = {r["chunk_id"] for r in filtered}
        merged = list(filtered)
        for r in unfiltered:
            if r["chunk_id"] not in seen:
                merged.append(r)
                seen.add(r["chunk_id"])
        return {"retrieved_chunks": merged[:top_k]}
    else:
        return {"retrieved_chunks": search_fn(query, top_k=top_k)}


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

    # Parse [Source N] and [Source N, Source M, ...] references (deduplicated, first-appearance order)
    seen = set()
    cited_indices = []
    for block in re.finditer(r"\[Sources?\s+[^\]]+\]", response.content):
        for m in re.findall(r"(\d+)", block.group()):
            idx = int(m)
            if idx not in seen:
                seen.add(idx)
                cited_indices.append(idx)

    # Build citations in first-appearance order
    citations = []
    for idx in cited_indices:
        i = idx - 1  # [Source N] is 1-indexed, chunks list is 0-indexed
        if 0 <= i < len(chunks):
            chunk = chunks[i]
            citations.append({
                "document_name": chunk["document_name"],
                "document_date": chunk["document_date"],
                "section_header": chunk["section_header"],
                "chunk_id": chunk["chunk_id"],
                "relevance_score": chunk["relevance_score"],
                "quoted_excerpt": chunk["chunk_text"][:200],
                "source_index": idx,
            })

    return {"answer": response.content, "citations": citations}


def validate_citations(state: AgentState) -> dict:
    """Verify each citation maps to an actual retrieved chunk.

    Drops any citation whose chunk_id is not in retrieved_chunks
    (guards against hallucinated source references or state corruption).
    """
    chunk_lookup = {
        c["chunk_id"]: c["chunk_text"]
        for c in state.get("retrieved_chunks", [])
    }

    validated = []
    for citation in state.get("citations", []):
        chunk_text = chunk_lookup.get(citation["chunk_id"])
        if chunk_text is None:
            logger.warning("Dropping citation with unknown chunk_id: %s", citation["chunk_id"])
            continue
        # Ensure the excerpt is an actual prefix of the chunk text
        citation["quoted_excerpt"] = chunk_text[:200]
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

    # Remap [Source N] references in answer text to sequential [N] numbering
    source_map = {}
    for new_idx, c in enumerate(citations, 1):
        orig_idx = c.get("source_index")
        if orig_idx is not None:
            source_map[orig_idx] = new_idx

    def _remap_source_block(match):
        block = match.group(0)
        nums = re.findall(r"(\d+)", block)
        remapped = [str(source_map[int(n)]) for n in nums if int(n) in source_map]
        if remapped:
            return "[" + ", ".join(remapped) + "]"
        return block

    answer = re.sub(r"\[Sources?\s+[^\]]+\]", _remap_source_block, answer)

    # Grounded answer with sources
    sources = "\n".join(
        f"  [{i+1}] {c['document_name']}, {c['document_date']}, "
        f"§{c['section_header']} (chunk {c['chunk_id'][:8]})"
        for i, c in enumerate(citations)
    )
    formatted = f"{answer}\n\nSources:\n{sources}" if sources else answer
    return {"answer": formatted}
