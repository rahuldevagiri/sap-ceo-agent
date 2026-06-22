from pathlib import Path

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

COLLECTION_NAME = "sap_intelligence"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

# Company profile shown in the dashboard "Company Overview" section.
COMPANY_INDUSTRY = {
    "SAP": "Enterprise Software & Cloud",
    "Oracle": "Enterprise Software & Cloud",
    "Salesforce": "Enterprise Software & CRM",
    "Microsoft": "Software & Cloud",
    "Workday": "Enterprise HR & Finance Software",
    "ServiceNow": "Enterprise Workflow Software",
}
DEFAULT_INDUSTRY = "Technology"

# Number of chunks retrieved per view in the multi-view RAG retriever.
# 4 views * RETRIEVAL_TOP_K chunks gives the breadth the dashboard analyzes
# per run; raise it to surface more of the indexed corpus.
RETRIEVAL_TOP_K = 40

LLM_PROVIDER = "ollama"
# A 3B model fits entirely in the 4 GB GTX 1650 VRAM, so Ollama does not need a
# large CPU/CUDA_Host offload buffer (which fails to allocate under memory
# pressure with the 4.68 GB 7B model). This keeps the LLM in `llm` mode instead
# of falling back, and runs much faster (full GPU offload).
LLM_MODEL = "qwen2.5:3b-instruct"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 1200
# Context window for the model. 8192 leaves ample room for the prompt AND the
# JSON response (qwen2.5:3b supports up to 32k); still fits the 4 GB GPU.
LLM_NUM_CTX = 8192
# Keep the model loaded in VRAM between requests so it isn't cold-reloaded.
LLM_KEEP_ALIVE = "30m"

SAP_SOURCES = {
    "company_rss": [
        "https://news.sap.com/feed/",
    ],
    "news_search_terms": [
        "SAP AI",
        "SAP cloud",
        "SAP earnings",
        "SAP partnership",
        "SAP competitor enterprise software",
    ],
    "competitors": ["Oracle", "Salesforce", "Microsoft", "Workday", "ServiceNow"]
}

# Collection volume targets. The collectors aim to gather at least this many
# unique documents across all sources before dedup/cleaning trims the set.
MIN_TARGET_DOCUMENTS = 100

