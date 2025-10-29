from langchain.chat_models import init_chat_model
from rag_app.settings import settings

def create_llm(model: str | None = None, provider: str = "openai"):
    return init_chat_model(model=model or settings.openai_model, model_provider=provider)