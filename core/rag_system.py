"""Main RAG system orchestrator with HDBSCAN super-chunk clustering"""

import streamlit as st
from typing import List, Tuple, Generator

from core.pdf_processor import PDFProcessor
from core.text_chunker import TextChunker
from core.embedding_manager import EmbeddingManager
from core.answer_generator import AnswerGenerator
from core.chunk_clusterer import ChunkClusterer, SuperChunk

from retreival.vector_store import VectorStore
from models.document_chunk import DocumentChunk
from config import MODEL_OPTIONS


class RAGSystem:
    """Main RAG system orchestrator with HDBSCAN semantic clustering"""

    def __init__(self, model_name: str, max_cluster_size: int = 5):
        self.embedding_manager = EmbeddingManager(model_name)
        self.vector_store = VectorStore()
        self.answer_generator = AnswerGenerator()
        self.chunk_clusterer = ChunkClusterer(max_cluster_size=max_cluster_size)

        self.base_chunks = []
        self.superchunks = []
        self.is_ready = False

    def setup_system(self, uploaded_file) -> bool:
        """Setup the complete RAG system with HDBSCAN super-chunk clustering"""
        if not uploaded_file:
            return False

        if not self.embedding_manager.initialize():
            st.error("Failed to initialize embedding model")
            return False

        processor = PDFProcessor()
        chunker = TextChunker()

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 1: Extract text from PDF
        status_text.text("📄 Extracting text from PDF...")
        doc_name = uploaded_file.name
        all_pages = processor.extract_text_from_pdf(uploaded_file, doc_name)

        if not all_pages:
            st.error("No text extracted from PDF")
            return False

        progress_bar.progress(0.2)

        # Step 2: Create base chunks (200-300 tokens each)
        status_text.text("✂️ Creating base text chunks...")
        self.base_chunks = chunker.create_chunks(all_pages)

        if not self.base_chunks:
            st.error("No chunks created")
            return False

        progress_bar.progress(0.35)

        # Step 3: Generate embeddings for base chunks
        status_text.text(
            f"🧠 Generating base embeddings with {MODEL_OPTIONS[self.embedding_manager.model_name]}..."
        )

        def update_progress_base(progress):
            progress_bar.progress(0.35 + (progress * 0.15))

        base_embeddings = self.embedding_manager.generate_embeddings(
            self.base_chunks, update_progress_base
        )
        progress_bar.progress(0.5)

        # Step 4: Create super-chunks via HDBSCAN clustering
        status_text.text("🔗 Clustering chunks with HDBSCAN...")
        self.superchunks = self.chunk_clusterer.create_superchunks(
            self.base_chunks, base_embeddings
        )

        st.info(
            f"✅ Created {len(self.superchunks)} super-chunks from {len(self.base_chunks)} base chunks"
        )
        progress_bar.progress(0.6)

        # Step 5: Generate embeddings for super-chunks
        status_text.text(
            f"🧠 Generating super-chunk embeddings with {MODEL_OPTIONS[self.embedding_manager.model_name]}..."
        )

        def update_progress_super(progress):
            progress_bar.progress(0.6 + (progress * 0.25))

        superchunk_embeddings = self.embedding_manager.generate_superchunk_embeddings(
            self.superchunks, update_progress_super
        )
        progress_bar.progress(0.85)

        # Step 6: Setup vector store with super-chunks
        status_text.text("🔍 Setting up vector search...")
        self.vector_store.build_superchunk_index(
            superchunk_embeddings, self.superchunks
        )
        progress_bar.progress(1.0)

        status_text.text("✅ System ready!")
        self.is_ready = True
        return True

    def query(
        self, user_query: str, api_key: str, top_k: int
    ) -> Tuple[List[Tuple[SuperChunk, float]], str]:
        """Process a user query and return super-chunk results"""
        if not self.is_ready:
            return [], "System not ready. Please upload and process documents first."

        # Rewrite query for better retrieval
        optimized_query = self.answer_generator.rewrite_query_with_gpt(
            user_query, api_key
        )

        # Get query embedding
        query_embedding = self.embedding_manager.encode_query(optimized_query)
        print(f"Query Embedding Shape: {query_embedding.shape}")
        # Retrieve relevant super-chunks
        retrieved_superchunks = self.vector_store.search(query_embedding, top_k)

        return retrieved_superchunks, optimized_query

    def generate_answer_stream(
        self,
        query: str,
        retrieved_superchunks: List[Tuple[SuperChunk, float]],
        api_key: str,
    ) -> Generator[str, None, None]:
        """Generate streaming answer from super-chunks"""
        # Convert super-chunks to format expected by answer generator
        formatted_chunks = []
        for superchunk, score in retrieved_superchunks:
            # Create a pseudo-chunk with page numbers list
            pseudo_chunk = DocumentChunk(
                text=superchunk.text,
                doc_name=superchunk.doc_name,
                page_num=superchunk.page_numbers[0],  # Keep for compatibility
                chunk_id=superchunk.superchunk_id,
            )
            # Add page_numbers attribute for the generator to use
            pseudo_chunk.page_numbers = superchunk.page_numbers
            formatted_chunks.append((pseudo_chunk, score))

        yield from self.answer_generator.generate_answer_stream(
            query, formatted_chunks, api_key
        )

    def get_system_stats(self) -> dict:
        """Get system statistics"""
        if not self.is_ready:
            return {}

        docs = {}
        for chunk in self.base_chunks:
            docs[chunk.doc_name] = docs.get(chunk.doc_name, 0) + 1

        return {
            "total_base_chunks": len(self.base_chunks),
            "total_chunks": len(self.superchunks),
            "total_superchunks": len(self.superchunks),
            "total_documents": len(docs),
            "documents": docs,
            "model_name": self.embedding_manager.model_name,
            "device": self.embedding_manager.device,
            "avg_chunks_per_superchunk": (
                round(len(self.base_chunks) / len(self.superchunks), 2)
                if self.superchunks
                else 0
            ),
        }
