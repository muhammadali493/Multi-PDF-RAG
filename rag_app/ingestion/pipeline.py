from typing import List
from langchain_core.documents import Document

def enrich_metadata(docs: List[Document], filename: str, file_hash: str) -> List[Document]:
    for i, d in enumerate(docs):
        md = dict(d.metadata or {})
        md["source"] = filename
        md["file_hash"] = file_hash
        md["chunk_id"] = f"{file_hash}-{i}"
        d.metadata = md
    return docs