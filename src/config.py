import os
from dotenv import load_dotenv

load_dotenv(override=True)


class Config:
    APP_USERNAME: str = os.getenv("APP_USERNAME", "")
    APP_PASSWORD: str = os.getenv("APP_PASSWORD", "")

    OSS_MODEL_ID: str = os.getenv("OSS_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
    OSS_DEVICE: str = os.getenv("OSS_DEVICE", "auto")
    OSS_MAX_NEW_TOKENS: int = int(os.getenv("OSS_MAX_NEW_TOKENS", "1024"))
    OSS_TEMPERATURE: float = float(os.getenv("OSS_TEMPERATURE", "0.4"))

    FRONTIER_PROVIDER: str = os.getenv("FRONTIER_PROVIDER", "openai")
    FRONTIER_MAX_TOKENS: int = int(os.getenv("FRONTIER_MAX_TOKENS", "5000"))
    FRONTIER_TEMPERATURE: float = float(os.getenv("FRONTIER_TEMPERATURE", "0.4"))

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5-nano")
    JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "gpt-5.4-mini")

    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    MEMORY_TOP_K: int = int(os.getenv("MEMORY_TOP_K", "5"))
    RECENT_TURN_PAIRS: int = int(os.getenv("RECENT_TURN_PAIRS", "3"))

    DB_PATH: str = os.getenv("DB_PATH", "data/app.db")
    FAISS_DIR: str = os.getenv("FAISS_DIR", "data/faiss")


cfg = Config()
