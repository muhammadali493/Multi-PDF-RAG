from typing import List, Iterable, Optional, Dict, Any, Set
from langchain_core.documents import Document

def _unique_key_for_doc(doc: Document) -> str:
    # Prefer stable chunk id if present, else fallback to (source + content snippet)
    md = doc.metadata or {}
    if "chunk_id" in md:
        return md["chunk_id"]
    source = md.get("source", "")
    content_snippet = (doc.page_content or "")[:200]
    return f"{source}::{content_snippet}"

def parse_paraphrases(text: str, max_results: int) -> List[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # If the LLM returns a single paragraph, try splitting by punctuation or semicolon as a fallback
    if len(lines) == 1 and len(lines[0].split(".")) > 1:
        parts = [p.strip() for p in lines[0].split(".") if p.strip()]
        lines = parts
    return lines[:max_results]

def generate_queries_via_llm(llm, question: str, n: int = 4) -> List[str]:
    """
    Use the provided llm to generate up to `n` paraphrases of `question`.
    Expect the llm to accept a plain string and return a string (common for chat models).
    """
    instruction = (
        f"Generate {n} concise paraphrases (different phrasings / angles) of the user's question.\n"
        "Return each paraphrase on its own line, no numbering, keep them short.\n\n"
        f"Question: {question}\n\nParaphrases:"
    )
    # The llm passed in your app is the same object used for final answer; invoke it here.
    # The invocation method used in many LangChain setups is `invoke` or calling the runnable directly;
    # this code calls the llm as a callable (llm may be a Runnable in your setup).
    resp = llm.invoke(instruction) if hasattr(llm, "invoke") else llm(instruction)
    text = resp if isinstance(resp, str) else str(resp)
    return parse_paraphrases(text, n)

def multi_query_retrieve(
    base_retriever,
    llm,
    question: str,
    n_queries: int = 4,
    k_per_query: int = 4,
    where: Optional[Dict[str, Any]] = None
) -> List[Document]:
    """
    Run multi-query retrieval:
    - generate paraphrases with the llm
    - run the base_retriever for each paraphrase
    - deduplicate results (by chunk id or content)
    - return a list of unique Documents
    """
    # 1) generate queries
    queries = generate_queries_via_llm(llm, question, n=n_queries)
    print(f'Similar Queries to user query are: {queries}')
    if not queries:
        queries = [question]

    unique: Dict[str, Document] = {}
    seen: Set[str] = set()

    # 2) retrieve for each paraphrase
    for q in queries:
        # Many LangChain retrievers expose get_relevant_documents(query)
        if hasattr(base_retriever, "get_relevant_documents"):
            docs = base_retriever.get_relevant_documents(q)
        else:
            # Fallback: try calling as a callable (some retrievers are Runnable)
            docs = base_retriever(q)

        # If the retriever returns more than needed, slice to k_per_query
        docs = list(docs)[:k_per_query]

        for d in docs:
            key = _unique_key_for_doc(d)
            if key not in seen:
                seen.add(key)
                unique[key] = d

    # 3) return unique documents (preserve insertion order from queries)
    return list(unique.values())