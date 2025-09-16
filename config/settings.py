# config/settings.py
from pydantic_settings import BaseSettings
from typing import List
from .constants import MAX_FILE_SIZE, MAX_TOTAL_SIZE, ALLOWED_TYPES
from pydantic_settings import BaseSettings
# (optionally switch to the v2 style; keeping your current style is fine)

class Settings(BaseSettings):
    # ── Required ────────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str

    # ── Embeddings / Vector store ──────────────────────────────────────────────
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    CHROMA_DB_PATH: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "documents"

    # ── Retrieval knobs ────────────────────────────────────────────────────────
    VECTOR_SEARCH_K: int = 10                       # vector retriever k
    HYBRID_RETRIEVER_WEIGHTS: List[float] = [0.4, 0.6]
    RETRIEVAL_TOPK_PER_CLAIM: int = 4               # EvidenceGatherer k
    VERIFY_PASSAGES_PER_CLAIM: int = 3              # passages fed to verifier
    EVIDENCE_RECENCY_YEARS: int = 10                # soft recency heuristic

    # ── LLM models / tokens / temps ────────────────────────────────────────────
    OPENAI_CLAIM_MODEL: str = "gpt-4o-mini"
    OPENAI_VERIFY_MODEL: str = "gpt-4o-mini"
    OPENAI_REWRITER_MODEL: str = "gpt-4o-mini"

    CLAIM_MAX_TOKENS: int = 600
    VERIFY_MAX_TOKENS: int = 400
    REWRITE_MAX_TOKENS: int = 900

    TEMPERATURE_CLAIM: float = 0.0
    TEMPERATURE_VERIFY: float = 0.0
    TEMPERATURE_REWRITE: float = 0.2

    # ── Risk / policy / compliance ─────────────────────────────────────────────
    ENABLE_PHI_CHECK: bool = True
    ALLOWLIST_DOMAINS: List[str] = [
        "pubmed.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "clinicaltrials.gov",
        "dailymed.nlm.nih.gov",
        "cdc.gov",
        "who.int",
        "cochranelibrary.com",
        "fda.gov",
    ]
    DISCLAIMER_TEXT: str = (
        "This note is informational and not a substitute for clinical judgment "
        "or patient-specific guidance."
    )

    # ── Map fan-out / operational controls ─────────────────────────────────────
    VERIFY_MAP_CONCURRENCY: int = 8                 # soft parallelism hint
    MAX_CLAIMS_PER_BATCH: int = 32
    LLM_RETRY_COUNT: int = 1
    LLM_TIMEOUT_SECONDS: int = 60

    # ── Upload constraints / logging / cache ───────────────────────────────────
    MAX_FILE_SIZE: int = MAX_FILE_SIZE
    MAX_TOTAL_SIZE: int = MAX_TOTAL_SIZE
    ALLOWED_TYPES: List[str] = ALLOWED_TYPES
    LOG_LEVEL: str = "INFO"
    CACHE_DIR: str = "document_cache"
    CACHE_EXPIRE_DAYS: int = 7

    SKIP_SCOPE_GATE: bool = True
    class Config:
         env_file = ".env"
         env_file_encoding = "utf-8"
         extra = "ignore"   # ← optional but recommended to avoid future 'extra_forbidden'



settings = Settings()


    # class Config:
    #     env_file = ".env"
    #     env_file_encoding = "utf-8"

