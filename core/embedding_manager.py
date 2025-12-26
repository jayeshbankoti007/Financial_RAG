import streamlit as st
import numpy as np
import math
from sentence_transformers import SentenceTransformer
from typing import List
from tqdm import tqdm
from models.document_chunk import DocumentChunk
from config import MODEL_OPTIONS, BATCH_SIZE
import torch


class EmbeddingManager:
    """Handles embedding model loading and generation"""

    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model = None

    def _get_device(self, requested_device: str) -> str:
        """Determine the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    @st.cache_resource(show_spinner=False)
    def load_model(_self, model_name: str, device: str):
        """Load embedding model with caching - only loads once per session"""
        return SentenceTransformer(MODEL_OPTIONS[model_name], device=device)

    def initialize(self):
        """Initialize the embedding model (uses cache, so safe to call multiple times)"""
        if self.model is None:
            self.model = self.load_model(self.model_name, self.device)
        return self.model is not None

    def format_text_for_model(self, text: str, is_query: bool = False) -> str:
        """Format text based on model requirements"""
        if self.model_name.startswith("E5"):
            prefix = "query: " if is_query else "passage: "
            return f"{prefix}{text.strip()}"
        return text.strip()

    def generate_embeddings(
        self, chunks: List[DocumentChunk], progress_callback=None
    ) -> np.ndarray:
        """Generate embeddings for document chunks"""
        # Format texts for the specific model
        texts = [self.format_text_for_model(chunk.text) for chunk in chunks]

        embedding_dim = self.model.get_sentence_embedding_dimension()
        embeddings = np.zeros((len(texts), embedding_dim), dtype="float32")

        batch_size = BATCH_SIZE
        num_batches = math.ceil(len(texts) / batch_size)

        for batch_idx in tqdm(range(num_batches), desc="chunk embeddings"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(texts))

            batch_texts = texts[start_idx:end_idx]

            # Encode and ensure numpy array
            batch_embeddings = self.model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,  # Ensures numpy output
            )
            embeddings[start_idx:end_idx] = batch_embeddings

            if progress_callback:
                progress_callback((batch_idx + 1) / num_batches)

        # Assign embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query"""
        formatted_query = self.format_text_for_model(query, is_query=True)
        print(f"Encoding Query: {formatted_query}")
        output_embedding = self.model.encode(
            [formatted_query], normalize_embeddings=True, convert_to_numpy=True
        )

        return output_embedding

    def generate_superchunk_embeddings(
        self, superchunks: List, progress_callback=None
    ) -> np.ndarray:
        """
        Generate embeddings for super-chunks

        Args:
            superchunks: List of SuperChunk objects
        Returns:
            Numpy array of embeddings
        """
        texts = [superchunk.text for superchunk in superchunks]
        batch_size = BATCH_SIZE
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="superchunk embeddings"):
            batch = texts[i : i + batch_size]
            embeddings = self.model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            all_embeddings.append(embeddings)

            if progress_callback:
                progress = (i + len(batch)) / len(texts)
                progress_callback(progress)

        return np.vstack(all_embeddings)
