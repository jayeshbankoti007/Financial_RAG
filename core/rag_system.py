"""Main RAG system orchestrator"""
import streamlit as st
from typing import List, Tuple, Generator
from core.pdf_processor import PDFProcessor
from core.text_chunker import TextChunker
from core.embedding_manager import EmbeddingManager
from retreival.vector_store import VectorStore
from core.answer_generator import AnswerGenerator
from models.document_chunk import DocumentChunk
from config import MODEL_OPTIONS

class RAGSystem:
    """Main RAG system orchestrator"""    
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.embedding_manager = EmbeddingManager(model_name, device)
        self.vector_store = VectorStore()
        self.answer_generator = AnswerGenerator()
        self.chunks = []
        self.is_ready = False
    
    def setup_system(self, uploaded_file) -> bool:
        """Setup the complete RAG system"""
        if not uploaded_file:
            return False
        
        # Initialize embedding model
        if not self.embedding_manager.initialize():
            st.error("Failed to initialize embedding model")
            return False
        
        processor = PDFProcessor()
        chunker = TextChunker(self.model_name)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Extract text from PDF
        status_text.text("📄 Extracting text from PDF...")
        doc_name = uploaded_file.name        
        all_pages = processor.extract_text_from_pdf(uploaded_file, doc_name)

        if not all_pages:
            st.error("No text extracted from PDF")
            return False

        progress_bar.progress(0.25)
        
        # Create chunks
        status_text.text("✂️ Creating text chunks...")
        self.chunks = chunker.create_chunks(all_pages)
        progress_bar.progress(0.5)
        
        if not self.chunks:
            st.error("No chunks created")
            return False
        
        # Generate embeddings
        status_text.text(f"🧠 Generating embeddings with {MODEL_OPTIONS[self.embedding_manager.model_name]}...")
        
        def update_progress(progress):
            progress_bar.progress(0.5 + (progress * 0.25))
        
        embeddings = self.embedding_manager.generate_embeddings(self.chunks, update_progress)
        progress_bar.progress(0.75)
        
        # Setup vector store
        status_text.text("🔍 Setting up vector search...")
        self.vector_store.build_index(embeddings, self.chunks)
        progress_bar.progress(1.0)
        
        status_text.text("✅ System ready!")
        self.is_ready = True
        return True
    
    def query(self, user_query: str, api_key: str, top_k: int) -> Tuple[List[Tuple[DocumentChunk, float]], str]:
        """Process a user query and return results"""
        if not self.is_ready:
            return [], "System not ready. Please upload and process documents first."
        
        # Rewrite query for better retrieval
        if api_key:
            optimized_query = self.answer_generator.rewrite_query_with_gpt(user_query, api_key)
        else:
            optimized_query = user_query
        
        # Get query embedding
        query_embedding = self.embedding_manager.encode_query(optimized_query)
        
        # Retrieve relevant chunks
        retrieved_chunks = self.vector_store.search(query_embedding, top_k)
        
        return retrieved_chunks, optimized_query
    
    def generate_answer_stream(self, query: str, retrieved_chunks: List[Tuple[DocumentChunk, float]], 
                              api_key: str) -> Generator[str, None, None]:
        """Generate streaming answer"""
        yield from self.answer_generator.generate_answer_stream(query, retrieved_chunks, api_key)
    
    def generate_answer(self, query: str, retrieved_chunks: List[Tuple[DocumentChunk, float]], 
                       api_key: str) -> str:
        """Generate non-streaming answer"""
        if api_key:
            return self.answer_generator.generate_answer(query, retrieved_chunks, api_key)
        else:
            return self.answer_generator.simple_answer_extraction(query, retrieved_chunks)
    
    def get_system_stats(self) -> dict:
        """Get system statistics"""
        if not self.is_ready:
            return {}
        
        docs = {}
        for chunk in self.chunks:
            docs[chunk.doc_name] = docs.get(chunk.doc_name, 0) + 1
        
        return {
            'total_chunks': len(self.chunks),
            'total_documents': len(docs),
            'documents': docs,
            'model_name': self.embedding_manager.model_name,
            'device': self.embedding_manager.device
        }