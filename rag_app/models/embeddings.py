from langchain_openai import OpenAIEmbeddings
from rag_app.settings import settings

def create_embeddings(model: str | None = None):
    return OpenAIEmbeddings(model=model or settings.embedding_model)