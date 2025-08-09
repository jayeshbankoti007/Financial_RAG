"""Data models for the RAG system"""

from dataclasses import dataclass
import numpy as np
from typing import Optional

@dataclass
class DocumentChunk:
    """Class to store document chunks with metadata"""
    text: str
    doc_name: str
    page_num: int
    chunk_id: int
    embedding: Optional[np.ndarray] = None