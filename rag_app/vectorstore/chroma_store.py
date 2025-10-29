# filepath: d:\ML Projects\Langchain\rag_app\vectorstore\chroma_store.py
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List, Iterable, Optional, Dict, Any
from rag_app.settings import settings

class ChromaStore:
    def __init__(self, embeddings, persist_dir: str | None = None) -> None:
        self.store = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_dir or settings.chroma_dir
        )

    def add_documents(self, docs: Iterable[Document]) -> List[str]:
        return self.store.add_documents(list(docs))

    def similarity_search(self, query: str, k: int, where: Optional[Dict[str, Any]] = None) -> List[Document]:
        # LangChain passes 'filter' to vector stores; Chroma expects 'where' under the hood.
        return self.store.similarity_search(query, k=k, filter=where)

    def as_retriever(self, k: int, where: Optional[Dict[str, Any]] = None):
        kwargs = {"k": k}
        if where:
            kwargs["filter"] = where
        return self.store.as_retriever(search_kwargs=kwargs)