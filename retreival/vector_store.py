import os

# FORCE SINGLE THREAD - macOS compatibility
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import faiss
import numpy as np
from typing import List, Tuple
from models.document_chunk import DocumentChunk

# Set FAISS threading after import
faiss.omp_set_num_threads(4)


class VectorStore:
    """Handles FAISS HNSW vector store operations"""

    def __init__(self):
        self.index = None
        self.chunks = []

    def build_index(
        self,
        embeddings: np.ndarray,
        chunks: List[DocumentChunk],
        m: int = 32,
        ef_construction: int = 125,
        ef_search: int = 100,
    ):
        """Build HNSW FAISS index for cosine similarity"""
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("Embeddings cannot be empty")

        dim = embeddings.shape[1]

        # HNSW Index using Inner Product (works as cosine after normalization)
        self.index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)

        # Set construction + search params
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search

        self.index.add(embeddings.astype("float32"))
        self.chunks = chunks

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search for most relevant chunks"""
        if self.index is None or len(self.chunks) == 0:
            return []

        actual_top_k = min(top_k, len(self.chunks))

        scores, indices = self.index.search(
            query_embedding.astype("float32"), actual_top_k
        )
        print(f"FAISS Search Scores: {scores}")
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))

        return results

    def build_superchunk_index(
        self,
        embeddings: np.ndarray,
        superchunks: List,
        m: int = 32,
        ef_construction: int = 64,
        ef_search: int = 100,
    ):
        """HNSW index for superchunks"""
        dim = embeddings.shape[1]

        self.index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search

        self.index.add(embeddings.astype("float32"))
        self.chunks = superchunks
