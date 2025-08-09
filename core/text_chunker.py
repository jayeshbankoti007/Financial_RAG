"""Text chunking functionality"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
from models.document_chunk import DocumentChunk
from config import CHUNK_SIZE, CHUNK_OVERLAP

class TextChunker:
    """Handles text chunking with overlap"""
    
    def __init__(self, model_name, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        if model_name == "BAAI/bge-m3": 
            chunk_size = 1200
            overlap = 250            

        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )



    def create_chunks(self, pages_data: List[Dict]) -> List[DocumentChunk]:
        """Create document chunks from pages data"""
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