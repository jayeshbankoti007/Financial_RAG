"""Answer generation using OpenAI and fallback methods"""

import openai
import streamlit as st
from typing import List, Tuple, Set, Generator
from models.document_chunk import DocumentChunk
from config import (
    SYSTEM_PROMPT,
    QUERY_REWRITE_PROMPT,
    OPENAI_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_P,
)


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
                messages=[
                    {
                        "role": "user",
                        "content": QUERY_REWRITE_PROMPT.format(query=user_query),
                    }
                ],
                max_tokens=100,
                temperature=0.1,
            )

            rewritten_query = response.choices[0].message.content.strip()
            print(f"Original Query: {user_query}")
            print(f"Rewritten Query: {rewritten_query}")
            return rewritten_query

        except Exception as e:
            st.warning(f"Query rewriting failed: {str(e)}. Using original query.")
            return user_query

    def generate_answer_stream(
        self,
        query: str,
        retrieved_chunks: List[Tuple[DocumentChunk, float]],
        api_key: str,
    ) -> Generator[str, None, None]:
        """Generate answer using OpenAI GPT with streaming"""

        if not retrieved_chunks:
            yield "I cannot find relevant information in the provided documents to answer your question."
            return

        # Build context with page number lists
        context_parts = []
        for chunk, _ in retrieved_chunks:
            # Check if chunk has page_numbers list attribute (from super-chunks)
            if hasattr(chunk, "page_numbers"):
                page_info = f"Pages {chunk.page_numbers}"
            else:
                page_info = f"Page {chunk.page_num}"

            context_parts.append(f"{page_info}\nContent: {chunk.text}")

        context = "\n\n".join(context_parts)

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

                Provide a clear, accurate answer based on the context above. When referencing information, mention the specific page numbers from the context.""",
                },
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
