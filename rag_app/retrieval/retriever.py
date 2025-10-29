# filepath: d:\ML Projects\Langchain\rag_app\retrieval\retriever.py
from typing import List, Optional, Dict, Any

def build_source_filter(selected: List[str], all_files: List[str]) -> Optional[Dict[str, Any]]:
    if not selected:
        return None
    if "All files" in selected:
        return {"source": {"$in": all_files}}
    return {"source": {"$in": selected}}