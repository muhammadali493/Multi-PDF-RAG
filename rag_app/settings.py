#from pydantic import BaseSettings, Field
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    openai_model: str = Field("gpt-4o-mini", env="CHAT_MODEL")
    embedding_model: str = Field("text-embedding-3-large", env="EMBEDDING_MODEL")
    chroma_dir: str = Field("./chroma_db", env="CHROMA_DIR")
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")

    chunk_size: int = 1500
    chunk_overlap: int = 300
    top_k: int = 4
    top_k_broad: int = 10

    class Config:
        env_file = ".env"

settings = Settings()