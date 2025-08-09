"""Answer generation using OpenAI and fallback methods"""

import openai
import streamlit as st
import spacy
from typing import List, Tuple, Set, Generator
from models.document_chunk import DocumentChunk
from config import SYSTEM_PROMPT, QUERY_REWRITE_PROMPT, OPENAI_MODEL, MAX_TOKENS, TEMPERATURE, TOP_P

# Load spaCy model for fallback
nlp = spacy.load("en_core_web_sm")

class AnswerGenerator:
    """Handles answer generation with OpenAI and fallback methods"""
    
    def __init__(self):
        pass
    
    def rewrite_query_with_gpt(self, user_query: str, api_key: str) -> str:
        """Rewrite a user query to be concise and clear for embedding models"""
        try:
            client = openai.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{
                    "role": "user",
                    "content": QUERY_REWRITE_PROMPT.format(query=user_query)
                }],
                max_tokens=100,
                temperature=0.1
            )
            
            rewritten_query = response.choices[0].message.content.strip()
            print(f"Original Query: {user_query}")
            print(f"Rewritten Query: {rewritten_query}")
            return rewritten_query
            
        except Exception as e:
            st.warning(f"Query rewriting failed: {str(e)}. Using original query.")
            return user_query
    
    def generate_answer_stream(self, query: str, retrieved_chunks: List[Tuple[DocumentChunk, float]], 
                              api_key: str) -> Generator[str, None, None]:
        """Generate answer using OpenAI GPT with streaming"""
        
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
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user", 
                        "content": f"""Context from the financial document:
                    {context}
                    
                    Question: {query}
                    
                    Provide a clear, accurate answer based on the context above."""
                    }
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                stream=True 
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        except Exception as e:
            yield f"[OpenAI API Error: {str(e)}]"
    
    def generate_answer(self, query: str, retrieved_chunks: List[Tuple[DocumentChunk, float]], 
                       api_key: str) -> str:
        """Generate answer using OpenAI GPT (non-streaming)"""
        
        if not retrieved_chunks:
            return "I cannot find relevant information in the provided documents to answer your question."
        
        context = "\n\n".join([
            f"Source: {chunk.doc_name}, Page {chunk.page_num}\nContent: {chunk.text}"
            for chunk, _ in retrieved_chunks
        ])
        
        try:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user", 
                        "content": f"""Context from the financial document:
                    {context}
                    
                    Question: {query}
                    
                    Provide a clear, accurate answer based on the context above."""
                    }
                ],
                max_tokens=400,
                temperature=0.1,
                top_p=TOP_P
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            st.error(f"OpenAI API Error: {str(e)}")
            return self.simple_answer_extraction(query, retrieved_chunks)
    
    def simple_answer_extraction(self, query: str, retrieved_chunks: List[Tuple[DocumentChunk, float]]) -> str:
        """Fallback method for answer generation"""        
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
            answer += '. '.join(relevant_sentences[:3]) + '.'
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
    
    def generate_fallback_answer(self, query: str, retrieved_chunks: List[Tuple[DocumentChunk, float]]) -> str:
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
            sentence_scores.append((sentence, overlap/len(query_keywords)))

        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        relevant = [s for s, score in sentence_scores[:3] if score > 0]
        
        if relevant:
            answer += '. '.join(relevant) + '.'
        else:
            answer += best_chunk.text[:300] + '...'
        
        answer += f"\n\n*Generated using document extraction (no API)*"
        return answer