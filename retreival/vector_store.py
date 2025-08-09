"""Vector store functionality using FAISS"""

import faiss
import numpy as np
from typing import List, Tuple
from models.document_chunk import DocumentChunk

class VectorStore:
    """Handles FAISS vector store operations"""
    
    def __init__(self):
        self.index = None
        self.chunks = []
    
    def build_index(self, embeddings: np.ndarray, chunks: List[DocumentChunk]):
        """Build FAISS index from embeddings"""
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index.add(embeddings_normalized.astype('float32'))
        self.chunks = chunks
    
    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[DocumentChunk, float]]:
        """Search for most relevant chunks"""
        if not self.index:
            return []
        
        actual_top_k = min(top_k, len(self.chunks))
        
        scores, indices = self.index.search(query_embedding.astype('float32'), actual_top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks) and score > 0:
                results.append((self.chunks[idx], float(score)))
        
        return results