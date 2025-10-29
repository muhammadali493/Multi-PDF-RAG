from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_app.settings import settings
from typing import List
from langchain_core.documents import Document

def split_documents(pages: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(pages)