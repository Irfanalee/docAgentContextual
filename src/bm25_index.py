from rank_bm25 import BM25Okapi
from typing import List, Dict
import numpy as np

class BM25Index:
    def __init__(self) -> None:
        self.bm25 = None            # The BM25 index Object.
        self.documents = []         # Store original chunks
        self.tokenized_corpus =[]   # Store Tokenized versions

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization by splitting on whitespace and converting to lowercase.
    
        Args:
            text: Text to tokenize
        
        Returns:
            List of tokens
        """
        return text.lower().split()
    
    def add_docments(self, chunks: List[Dict]) -> None:
        """
        Add document chunks to the BM25 index.
        
        Args:
            chunks: List of chunks with 'chunk_text' and 'chunk_id' fields
        """
        self.documents = chunks

        # Tokenize corpus - combine context and chunk_text for better matching
        self.tokenized_corpus = []
        for chunk in chunks:

        
        