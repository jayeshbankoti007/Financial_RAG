import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import math
import streamlit as st
import fitz
import numpy as np
import faiss
import openai
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import re
from dataclasses import dataclass
import spacy
from typing import List, Set
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")


model_options = {
    "BAAI Base": "BAAI/bge-base-en-v1.5",
    "BAAI Large": "BAAI/bge-large-en-v1.5",
    "E5 Small": "intfloat/e5-small-v2",
    "E5 Base": "intfloat/e5-base-v2",
    "E5 Large": "intfloat/e5-large-v2",
    "E5 Multilingual Small": "intfloat/multilingual-e5-small",
    "E5 Multilingual Base": "intfloat/multilingual-e5-base",
    "E5 Multilingual Large": "intfloat/multilingual-e5-large",
    "E5 Multilingual LArge Instruct": "intfloat/multilingual-e5-large-instruct",
}


selected_model_name = st.selectbox(
    "Select embedding model:",
    options=list(model_options.keys()),
    index=0
)

device = "cuda" if st.checkbox("Use GPU if available", value=True) and torch.cuda.is_available() else "cpu"

top_k = st.slider("RAG choices to consider: More is better but costlier", min_value=3, max_value=30, value=7, step=1)

@dataclass
class DocumentChunk:
    """Class to store document chunks with metadata"""
    text: str
    doc_name: str
    page_num: int
    chunk_id: int
    embedding: np.ndarray = None

class PDFProcessor:
    """Handles PDF text extraction and cleaning"""
    
    def __init__(self):
        self.documents = []
    
    def extract_text_from_pdf(self, pdf_file, doc_name: str) -> List[Dict]:
        """Extract text from PDF with metadata"""
        try:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            pages_data = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # Basic cleaning
                cleaned_text = self.clean_text(text)
                
                if cleaned_text.strip():
                    pages_data.append({
                        'text': cleaned_text,
                        'doc_name': doc_name,
                        'page_num': page_num + 1
                    })
            
            doc.close()
            return pages_data
            
        except Exception as e:
            st.error(f"Error processing PDF {doc_name}: {str(e)}")
            return []
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\/\%\$\&\+\=\@\#]', ' ', text)
        return text.strip()

    
class TextChunker:
    """Handles text chunking with overlap"""
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
    
    def create_chunks(self, pages_data: List[Dict]) -> List[DocumentChunk]:
        chunks = []
        chunk_id = 0
        
        for page_data in pages_data:
            text = page_data['text']
            doc_name = page_data['doc_name']
            page_num = page_data['page_num']
            
            page_chunks = self.splitter.split_text(text)
            for chunk_text in page_chunks:
                chunks.append(DocumentChunk(
                    text=chunk_text.strip(),
                    doc_name=doc_name,
                    page_num=page_num,
                    chunk_id=chunk_id
                ))
                chunk_id += 1
        
        return chunks
    

class RAGSystem:
    """Main RAG system orchestrator"""
    
    def __init__(self):
        self.embedding_model = None
        self.vector_store = None
        self.chunks = []
        self.is_ready = False
    
    @st.cache_resource
    def load_embedding_model(_self):
        """Load E5 embedding model"""
        embedding_model = SentenceTransformer(model_options[selected_model_name], device=device)        
        return embedding_model
    
    def setup_system(self, uploaded_file):
        """Setup the complete RAG system"""
        if not uploaded_file:
            return False
        
        # Initialize embedding model
        self.embedding_model = self.load_embedding_model()
        if not self.embedding_model:
            return False
        
        processor = PDFProcessor()
        chunker = TextChunker()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Extract text from all PDFs
        status_text.text("📄 Extracting text from PDFs...")

        doc_name = uploaded_file.name        
        all_pages = processor.extract_text_from_pdf(uploaded_file, doc_name)

        if not all_pages:
            st.error("No text extracted from PDFs")
            return False

        progress_bar.progress(0.25)
    

        # Create chunks
        status_text.text("✂️ Creating text chunks for the extracted text...")
        self.chunks = chunker.create_chunks(all_pages)
        progress_bar.progress(0.5)
        
        if not self.chunks:
            st.error("No chunks created")
            return False
        
        # Generate embeddings
        status_text.text("🧠 Generating embeddings with E5 small for each chunks...")
        
        def format_for_e5(chunk_text: str) -> str:
            return f"passage: {chunk_text.strip()}"
            
        if selected_model_name[:2] == "E5":
            texts = [format_for_e5(chunk.text) for chunk in self.chunks]
        else:
            texts = [chunk.text for chunk in self.chunks]

        embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        embeddings = np.zeros((len(texts), embedding_dim), dtype='float32')

        batch_size = 32
        num_batches = math.ceil(len(texts) / batch_size)

        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(texts))
            
            batch_texts = texts[start_idx:end_idx]
            
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                show_progress_bar=False,
                normalize_embeddings=True
            )
            embeddings[start_idx:end_idx] = batch_embeddings
            progress_bar.progress((batch_idx + 1) / num_batches)

        status_text.text("✅ Embeddings generated successfully!")

        for chunk, embedding in zip(self.chunks, embeddings):
            chunk.embedding = embedding

        progress_bar.progress(0.75)
        
        # Setup FAISS vector store
        status_text.text("🔍 Setting up vector search...")
        
        self.setup_vector_store(embeddings)
        progress_bar.progress(1.0)
        
        status_text.text("✅ System ready!")
        self.is_ready = True

        return True


    def rewrite_query_with_gpt(self, user_query: str, api_key: str) -> str:
        """
        Rewrite a user query to be concise and clear for embedding models,
        without adding new facts or assumptions.
        """
        client = openai.OpenAI(api_key=api_key)

        prompt = f"""
            You are a financial analyst assistant. You need to rewrite the following search query so it is concise specific,
            without changing its meaning or removing important terms. This improved query will be used for vector search in the next step. 
            If an acronym is given, keep both the acronym and its full form if present.
            Do not add new facts, names, or assumptions.
            Output only the rewritten query.

            Query: {user_query}
        """

        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt
        )

        rewritten_query = response.output_text.strip()
        print(f"Earlier Query: {user_query} \n Rewritten Query: {rewritten_query} ")
        return rewritten_query


    def setup_vector_store(self, embeddings: np.ndarray):
        """Setup FAISS vector store"""
        dimension = embeddings.shape[1]
        self.vector_store = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.vector_store.add(embeddings_normalized.astype('float32'))
    

    def retrieve_chunks(self, query: str, api_key) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve most relevant chunks for a query"""
        if not self.is_ready:
            return []
        
        optimised_query = self.rewrite_query_with_gpt(query, api_key)

        if selected_model_name[:2] == "E5":
            optimised_query = f"query: {optimised_query}"

        query_embedding = self.embedding_model.encode([optimised_query], normalize_embeddings=True)        

        # Search vector store
        scores, indices = self.vector_store.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            # if idx < len(self.chunks):  # Ensure valid index
            results.append((self.chunks[idx], float(score)))
        
        return results
    

    def generate_answer_with_openai_stream(self, query: str, retrieved_chunks: List[Tuple[DocumentChunk, float]], api_key: str):
        """Generate answer using OpenAI GPT-4o-mini with streaming."""

        if not retrieved_chunks:
            yield "I cannot find relevant information in the provided documents to answer your question."
            return
        
        context = "\n\n".join([
            f"Page {chunk.page_num}\nContent: {chunk.text}"
            for chunk, _ in retrieved_chunks
        ])
        
        try:
            client = openai.OpenAI(api_key=api_key)
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "system",
                    "content": """You are a financial analyst assistant. Your task is to answer questions based ONLY on the provided context from a financial document. 
                        You would be provided the most relevant chunks from the document. \n
                        IMPORTANT RULES:
                        1. Answer ONLY if the answer is based on information in the provided context chunks.
                        2. If the answer is not in the context, respond: "I cannot find this information in the provided document" .
                        3. Be specific and reference to all the page numbers used to answer the question.
                        4. Keep answers clear, concise, and professional.
                        5. Do not add information from your general knowledge, stick to the documents"""
                                        }, {
                            "role": "user", 
                            "content": f"""Context from the financial document:
                        {context}
                        Question: {query} . Provide a clear, accurate answer based on the context above."""
                }],
                max_tokens=800,
                temperature=0.2,
                top_p=0.9,
                stream=True 
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        except Exception as e:
            yield f"[OpenAI API Error: {str(e)}]"



    def generate_answer_with_openai(self, query: str, retrieved_chunks: List[Tuple[DocumentChunk, float]], api_key: str) -> str:
        """Generate answer using OpenAI GPT-4o-mini"""
        if not retrieved_chunks:
            return "I cannot find relevant information in the provided documents to answer your question."
        
        # Prepare context from retrieved chunks
        context = "\n\n".join([
            f"Source: {chunk.doc_name}, Page {chunk.page_num}\nContent: {chunk.text}"
            for chunk, _ in retrieved_chunks
        ])
        
        try:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "system",
                    "content": """You are a financial analyst assistant. Your task is to answer questions based ONLY on the provided context from a financial document. 
                        You would be provided the most relevant chunks from the document. \n
                        IMPORTANT RULES:
                        1. Answer ONLY if the answer is based on information in the provided context chunks.
                        2. If the answer is not in the context, respond: "I cannot find this information in the provided document" .
                        3. Be specific and reference to all the page numbers used to answer the question.
                        4. Keep answers clear, concise, and professional.
                        5. Do not add information from your general knowledge, stick to the documents"""
                                        }, {
                            "role": "user", 
                            "content": f"""Context from the financial document:
                        {context}
                        Question: {query} . Provide a clear, accurate answer based on the context above."""
                }],
                max_tokens=400,
                temperature=0.1,
                top_p=0.9
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            st.error(f"OpenAI API Error: {str(e)}")
            return self.simple_answer_extraction(query, retrieved_chunks)
    

    def simple_answer_extraction(self, query: str, retrieved_chunks: List[Tuple[DocumentChunk, float]]) -> str:
        """Fallback method for answer generation"""        
        # Simple keyword-based extraction
        best_chunk, _ = retrieved_chunks[0]
        
        answer = f"Based on the documents, here's what I found:\n\n"
        answer += f"**From {best_chunk.doc_name}, Page {best_chunk.page_num}:**\n\n"
        
        # Extract relevant sentences
        sentences = best_chunk.text.split('. ')
        query_words = set(query.lower().split())
        
        relevant_sentences = []
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            if len(query_words.intersection(sentence_words)) >= 1:
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            answer += '. '.join(relevant_sentences[:top_k]) + '.'
        else:
            answer += best_chunk.text[:400] + '...'
        
        answer += f"\n\n*Note: This answer was generated using document extraction (no API). For better answers, please provide an OpenAI API key.*"
        return answer


    def extract_keywords(self, text: str) -> Set[str]:
        """Extract important words (nouns, verbs, adjectives) from text"""
        doc = nlp(text.lower())
        
        keywords = set()
        for token in doc:
            # Keep important POS tags, skip stop words and punctuation
            if (token.pos_ in ['NOUN', 'VERB', 'ADJ'] and 
                not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 2):
                keywords.add(token.lemma_)  # Use lemma for better matching
        
        return keywords


    def generate_fallback_answer(self, query: str, retrieved_chunks: List):
        """Simple fallback with keyword overlap"""
        if not retrieved_chunks:
            return "No relevant documents found."
        
        # Extract keywords from query
        query_keywords = self.extract_keywords(query)
        best_chunk, _ = retrieved_chunks[0]
        
        answer = f"Based on the documents:\n\n"
        answer += f"**From {best_chunk.doc_name}, Page {best_chunk.page_num}:**\n\n"
        
        # Find best sentences by overlap
        sentences = [s.strip() for s in best_chunk.text.split('.') if len(s.strip()) > 20]
        
        sentence_scores = []

        for sentence in sentences:
            sentence_keywords = self.extract_keywords(sentence)
            if not query_keywords or not sentence_keywords:
                continue

            overlap = len(query_keywords.intersection(sentence_keywords))            
            sentence_scores.append(sentence , overlap/len(query_keywords))

        sentence_scores.sort(key=lambda x: x[1], reverse=True)

        relevant = [s for s, score in sentence_scores[:3] if score > 0]
        
        if relevant:
            answer += '. '.join(relevant) + '.'
        else:
            answer += best_chunk.text[:300] + '...'
        
        answer += f"\n\n*Generated using document extraction (no API)*"
        return answer


def main():
    st.set_page_config(
        page_title="Financial RAG Q&A System",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Financial Document Q&A System")
    st.markdown("**Powered by E5 embeddings + GPT-4o-mini**")
    
    # Initialize RAG system in session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    
    # Sidebar for API configuration and file upload
    with st.sidebar:
        st.header("🔑 OpenAI Configuration")
        
        # OpenAI API Key Input
        api_key = st.text_input(
            "Enter your OpenAI API Key Here",
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
            st.warning("No API Key Provided - Will respond using Spacy Text Extraction (no API)")
        
        st.divider()
        
        st.header("Document Upload")
        uploaded_files = st.file_uploader(
            "Add PDF file",
            type=['pdf'],
            accept_multiple_files=False,
            help="Upload your Financial Document (max 200MB)"
        )
        
        if uploaded_files and not st.session_state.rag_system.is_ready:
            if st.button("🚀 Process Documents", type="primary"):
                with st.spinner("Setting up RAG system..."):
                    success = st.session_state.rag_system.setup_system(uploaded_files)
                    if success:
                        st.success("✅ Documents processed successfully!")
                        st.rerun()
        
        # System status
        st.header("📊 System Status")
        if st.session_state.rag_system.is_ready:
            st.success("🟢 System Ready")
            
            # Show metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Chunks", len(st.session_state.rag_system.chunks))
            with col2:
                docs = {}
                for chunk in st.session_state.rag_system.chunks:
                    docs[chunk.doc_name] = docs.get(chunk.doc_name, 0) + 1
                st.metric("Documents", len(docs))
            
            # Document breakdown
            st.subheader("📄 Documents Loaded:")
            for doc_name, chunk_count in docs.items():
                st.write(f"• {doc_name}: {chunk_count} chunks")
                
        else:
            st.warning("🟡 Upload documents to get started")
        
        # Reset button
        if st.session_state.rag_system.is_ready:
            st.divider()
            if st.button("🔄 Reset System", help="Clear all data and start over"):
                del st.session_state.rag_system
                st.session_state.rag_system = RAGSystem()
                st.rerun()
    
    # Main Q&A interface
    if st.session_state.rag_system.is_ready:
        st.header("💬 Ask Questions About Your Documents")
        
        # # Sample questions
        st.subheader("💡 Try these sample questions:")
        sample_questions = [
            "How is revenue recognised under the accrual basis of accounting?",
            "What is GAAP?",
            "What are the main components of a balance sheet?", 
            "Describe the process of closing entries at the end of the accounting period.?",
            "Where is retained earnings shown on the financial statements?",
        ]
        
        cols = st.columns(len(sample_questions))
        for i, question in enumerate(sample_questions):
            with cols[i]:
                if st.button(question, key=f"sample_{i}", help="Click to use this question"):
                    st.session_state.current_query = question
        
        # Query input
        query = st.text_area(
            "Enter your question:",
            value=st.session_state.get('current_query', ''),
            placeholder="e.g., How is depreciation calculated in accounting?",
            height=100
        )
        

        if st.button("🔍 Get Answer", type="primary"):
            if query.strip():
                with st.spinner("🔍 Searching documents and generating answer..."):
                    # Retrieve relevant chunks
                    retrieved_chunks = st.session_state.rag_system.retrieve_chunks(query, st.session_state.api_key)
                    
                    if retrieved_chunks:
                        if st.session_state.get('api_key'):
                            st.subheader("📝 Answer")

                            placeholder = st.empty()
                            answer_text = ""

                            for token in st.session_state.rag_system.generate_answer_with_openai_stream(query, retrieved_chunks, st.session_state.api_key):
                                answer_text += token
                                placeholder.markdown(answer_text)
                                
                            # answer = st.session_state.rag_system.generate_answer_with_openai(
                            #     query, retrieved_chunks, st.session_state.api_key
                            # )
                        else:
                            answer = st.session_state.rag_system.simple_answer_extraction(
                                query, retrieved_chunks
                            )
                        
                            # Display answer
                            st.markdown(answer)
                        
                        # Display source chunks
                        st.subheader("📚 Source Information -- (Top 7)")
                        st.caption("These are the top 7 document sections used to generate the answer:")
                        
                        for i, (chunk, score) in enumerate(retrieved_chunks[:7]):
                            with st.expander(
                                f"📄 Source {i+1}: {chunk.doc_name}, Page {chunk.page_num} (Relevance: {score:.3f})",
                                expanded=(i == 0)
                            ):
                                st.write(chunk.text)
                                st.caption(f"Chunk ID: {chunk.chunk_id}")
                    else:
                        st.warning("❌ No relevant information found in the documents.")
            else:
                st.warning("Please enter a question before searching.")

        # Show query history
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        
        if query and query not in st.session_state.query_history:
            st.session_state.query_history.append(query)
        
        if st.session_state.query_history:
            with st.expander("📋 Query History"):
                for i, past_query in enumerate(reversed(st.session_state.query_history[-5:])):
                    st.write(f"{len(st.session_state.query_history)-i}. {past_query}")
    
    else:
        # Welcome screen with instructions
        st.markdown("---")
        st.header("🚀 Getting Started")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Instructions")
            st.markdown("""
            1. **Get OpenAI API Key** (recommended)
               - Visit https://platform.openai.com
               - Create account and add $5-10 credit
               - Generate API key \n
            
            2. **Upload PDFs**
               - Use sidebar to upload financial documents
               - Supports multiple PDFs up to 20MB each \n
            
            3. **Process Documents** 
               - Click "Process Documents" 
               - Wait 5-10 minutes for setup \n
            
            4. **Ask Questions**
               - Ask natural language questions
               - Get AI-powered answers with sources \n
            
            5. **Reset System**
               - Clear all data and start over 
            """)
        
        with col2:
            st.subheader("✨ Key Features")
            st.markdown("""
            - **E5 Embeddings**: State-of-the-art semantic search
            - **GPT-4o-mini**: High-quality answer generation
            - **Source Attribution**: Every answer shows exact sources
            - **Fallback Mode**: Works without API key and extracts text chunk which answers it best
            - **Multi-document**: Search across multiple PDFs
            - **Cost-effective**: ~$0.001 per question using 4o-mini
            """)
        
        st.markdown("---")
        st.info("💡 **Pro Tip**: For best results, use financial accounting textbooks, annual reports, or similar structured documents.")

if __name__ == "__main__":
    main()