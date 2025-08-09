"""PDF processing functionality"""

import fitz
import re
import streamlit as st
from typing import List, Dict

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