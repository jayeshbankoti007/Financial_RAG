"""Configuration settings for the RAG system"""

# Embedding model options
MODEL_OPTIONS = {
    "BAAI latest": "BAAI/bge-m3",
    "BAAI Base": "BAAI/bge-base-en-v1.5",
    "BAAI Large": "BAAI/bge-large-en-v1.5",
    "E5 Small": "intfloat/e5-small-v2",
    "E5 Base": "intfloat/e5-base-v2",
    "E5 Large": "intfloat/e5-large-v2",
    "E5 Multilingual Small": "intfloat/multilingual-e5-small",
    "E5 Multilingual Base": "intfloat/multilingual-e5-base",
    "E5 Multilingual Large": "intfloat/multilingual-e5-large",
    "E5 Multilingual Large Instruct": "intfloat/multilingual-e5-large-instruct",
}

# Text processing settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
BATCH_SIZE = 32

# OpenAI settings
OPENAI_MODEL = "gpt-4o-mini"
MAX_TOKENS = 800
TEMPERATURE = 0.2
TOP_P = 0.9

# Default values
DEFAULT_TOP_K = 7
MIN_TOP_K = 3
MAX_TOP_K = 30

# Sample questions for the UI
SAMPLE_QUESTIONS = [
    "How is revenue recognised under the accrual basis of accounting?",
    "What is GAAP?",
    "What are the main components of a balance sheet?", 
    "Describe the process of closing entries at the end of the accounting period.?",
    "Where is retained earnings shown on the financial statements?",
]


SYSTEM_PROMPT = """You are a financial analyst assistant. Your task is to answer questions based ONLY on the provided context from a financial document. 
You would be provided the most relevant chunks from the document.

IMPORTANT RULES:
1. Answer ONLY if the answer is based on information in the provided context chunks.
2. If the answer is not in the context, respond: "I cannot find this information in the provided document" .
3. Be specific and reference to all the page numbers used to answer the question.
4. Keep answers clear, concise, and professional.
5. Do not add information from your general knowledge, stick to the documents"""


QUERY_REWRITE_PROMPT = """You are a financial analyst assistant. You need to rewrite the following search query so it is concise specific,
without changing its meaning or removing important terms. This improved query will be used for vector search in the next step. 
If an acronym is given, keep both the acronym and its full form if present.
Do not add new facts, names, or assumptions.
Output only the rewritten query.

Query: {query}"""