"""Main Streamlit application for the RAG system"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import streamlit as st
from core.rag_system import RAGSystem
from config import MODEL_OPTIONS, SAMPLE_QUESTIONS, DEFAULT_TOP_K, MIN_TOP_K, MAX_TOP_K
from typing import List, Tuple


def initialize_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""

def render_sidebar():
    """Render the sidebar with configuration and upload"""
    with st.sidebar:
        st.header("🔑 Configuration")
        
        # Model selection
        selected_model_name = st.selectbox(
            "Select embedding model:",
            options=list(MODEL_OPTIONS.keys()),
            index=0
        )
        
        # Device selection
        device = "cuda" if st.checkbox("Use GPU if available", value=True) and torch.cuda.is_available() else "cpu"
        
        # Top-k slider
        top_k = st.slider(
            "RAG choices to consider: More is better but costlier", 
            min_value=MIN_TOP_K, 
            max_value=MAX_TOP_K, 
            value=DEFAULT_TOP_K, 
            step=1
        )
        
        # OpenAI API Key Input
        api_key = st.text_input(
            "Enter your OpenAI API Key",
            type="password",
            placeholder="sk-proj-...",
            help="Get your API key from https://platform.openai.com/api-keys"
        )
        st.session_state.api_key = api_key
        
        # API Status
        if api_key:
            st.success("OpenAI API Key Provided")
            st.info("Model Used: GPT-4o-mini")
        else:
            st.warning("No API Key - Will use text extraction fallback")
        
        st.divider()
        
        # File upload
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload PDF file",
            type=['pdf'],
            accept_multiple_files=False,
            help="Upload your Financial Document (max 200MB)"
        )
        
        # Process documents button
        if uploaded_file:
            if not st.session_state.rag_system or not st.session_state.rag_system.is_ready:
                if st.button("🚀 Process Document", type="primary"):
                    st.session_state.rag_system = RAGSystem(selected_model_name, device)
                    with st.spinner("Setting up RAG system..."):
                        success = st.session_state.rag_system.setup_system(uploaded_file)
                        if success:
                            st.success("✅ Document processed successfully!")
                            st.rerun()
        
        # System status
        render_system_status()
        
        return top_k

def render_system_status():
    """Render system status in sidebar"""
    st.header("📊 System Status")
    
    if st.session_state.rag_system and st.session_state.rag_system.is_ready:
        st.success("🟢 System Ready")
        
        # Show metrics
        stats = st.session_state.rag_system.get_system_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Chunks", stats['total_chunks'])
        with col2:
            st.metric("Documents", stats['total_documents'])
        
        # Document breakdown
        st.subheader("📄 Documents Loaded:")
        for doc_name, chunk_count in stats['documents'].items():
            st.write(f"• {doc_name}: {chunk_count} chunks")
        
        # Reset button
        st.divider()
        if st.button("🔄 Reset System", help="Clear all data and start over"):
            st.session_state.rag_system = None
            st.rerun()
            
    else:
        st.warning("🟡 Upload documents to get started")

def render_sample_questions():
    """Render clickable sample questions"""
    st.subheader("💡 Try these sample questions:")
    
    cols = st.columns(len(SAMPLE_QUESTIONS))
    for i, question in enumerate(SAMPLE_QUESTIONS):
        with cols[i]:
            if st.button(question, key=f"sample_{i}", help="Click to use this question"):
                st.session_state.current_query = question

def render_query_interface(top_k: int):
    """Render the main query interface"""
    st.header("💬 Ask Questions About Your Documents")
    
    render_sample_questions()
    
    # Query input
    query = st.text_area(
        "Enter your question:",
        value=st.session_state.get('current_query', ''),
        placeholder="e.g., How is depreciation calculated in accounting?",
        height=100
    )
    
    if st.button("🔍 Get Answer", type="primary"):
        if query.strip():
            process_query(query, top_k)
        else:
            st.warning("Please enter a question before searching.")
    
    # Show query history
    render_query_history()

def process_query(query: str, top_k: int):
    """Process a user query and display results"""
    with st.spinner("🔍 Searching documents and generating answer..."):
        # Get retrieved chunks
        retrieved_chunks, optimized_query = st.session_state.rag_system.query(
            query, st.session_state.api_key, top_k
        )
        
        if retrieved_chunks:
            # Generate and display answer
            if st.session_state.api_key:
                st.subheader("📝 Answer")
                placeholder = st.empty()
                answer_text = ""
                
                for token in st.session_state.rag_system.generate_answer_stream(
                    optimized_query, retrieved_chunks, st.session_state.api_key
                ):
                    answer_text += token
                    placeholder.markdown(answer_text)
            else:
                answer = st.session_state.rag_system.generate_answer(
                    optimized_query, retrieved_chunks, st.session_state.api_key
                )
                st.subheader("📝 Answer")
                st.markdown(answer)
            
            # Display source chunks
            render_source_information(retrieved_chunks)
            
            # Add to query history
            if query not in st.session_state.query_history:
                st.session_state.query_history.append(query)
        else:
            st.warning("❌ No relevant information found in the documents.")

def render_source_information(retrieved_chunks: List[Tuple]):
    """Render source information section"""
    st.subheader("📚 Source Information - Top 7")
    st.caption(f"Total of || {len(retrieved_chunks)} || document sections used to generate the answer")
    
    for i, (chunk, score) in enumerate(retrieved_chunks[:7]):
        with st.expander(
            f"📄Page {chunk.page_num} (Relevance: {score:.3f})",
            expanded=(i == 0)
        ):
            st.write(chunk.text)
            st.caption(f"Chunk ID: {chunk.chunk_id}")


def render_query_history():
    """Render query history section"""
    if st.session_state.query_history:
        with st.expander("📋 Query History"):
            for i, past_query in enumerate(reversed(st.session_state.query_history[-5:])):
                st.write(f"{len(st.session_state.query_history)-i}. {past_query}")

def render_welcome_screen():
    """Render welcome screen with instructions"""
    st.markdown("---")
    st.header("🚀 Getting Started")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Instructions")
        st.markdown("""
        1. **Get OpenAI API Key** (recommended)
           - Visit https://platform.openai.com
           - Create account and add $5-10 credit
           - Generate API key

        2. **Upload PDF**
           - Use sidebar to upload financial documents
           - Supports PDFs up to 200MB

        3. **Process Document** 
           - Click "Process Document" 
           - Wait for setup completion

        4. **Ask Questions**
           - Ask natural language questions
           - Get AI-powered answers with sources

        5. **Reset System**
           - Clear all data and start over 
        """)
    
    with col2:
        st.subheader("✨ Key Features")
        st.markdown("""
        - **Advanced Embeddings**: BAAI/E5 models for semantic search
        - **GPT-4o-mini**: High-quality answer generation
        - **Source Attribution**: Every answer shows exact sources
        - **Fallback Mode**: Works without API key
        - **Query Optimization**: AI-powered query rewriting
        - **Cost-effective**: ~$0.001 per question
        """)
    
    st.markdown("---")
    st.info("💡 **Pro Tip**: For best results, use financial accounting textbooks, annual reports, or similar structured documents.")

def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="Financial RAG Q&A System",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Financial Document Q&A System")
    st.markdown("**Powered by Advanced Embeddings + GPT-4o-mini**")
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar and get configuration
    top_k = render_sidebar()
    
    # Main content area
    if st.session_state.rag_system and st.session_state.rag_system.is_ready:
        render_query_interface(top_k)
    else:
        render_welcome_screen()

if __name__ == "__main__":
    main()