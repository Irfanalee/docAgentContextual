# Contextual Retrieval System

A Python implementation of Anthropic's Contextual Retrieval approach for improved document search and retrieval.

## 📚 Overview

This project implements the **Contextual Retrieval** technique described in [Anthropic's Engineering Blog](https://anthropic.com/news/contextual-retrieval), achieving up to **67% improvement** in retrieval accuracy through contextual embeddings and hybrid search.

## 🎯 Key Innovation

Traditional RAG systems chunk documents and embed them directly. This loses context. Our approach adds contextual descriptions to each chunk before embedding, dramatically improving retrieval accuracy.

### Traditional vs Contextual Retrieval

```
┌─────────────────────────────────────────────────────────────┐
│ TRADITIONAL RAG                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Document → Chunk → Embed → Store                          │
│                                                             │
│  Chunk: "Revenue grew 25% to $2.3M"                        │
│  ❌ Missing: Which company? Which quarter?                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ CONTEXTUAL RETRIEVAL (This Implementation)                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Document → Chunk → Add Context → Embed → Store            │
│                      ↓                                      │
│                   Claude API                                │
│                                                             │
│  Context: "This chunk is from ACME Corp's Q3 2024          │
│           financial report, discussing revenue growth"      │
│                                                             │
│  Chunk: "Revenue grew 25% to $2.3M"                        │
│  ✅ Now searchable with full context!                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    IMPLEMENTED (Phase 1)                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Document Loader                                              │
│     └─> Load PDF files → Extract text                           │
│                                                                  │
│  2. Chunker                                                      │
│     └─> Split text → 800 token chunks with 200 overlap          │
│                                                                  │
│  3. Contextualizer ⭐ (Key Innovation)                           │
│     └─> Claude API → Generate context for each chunk            │
│                                                                  │
│  4. Embedder                                                     │
│     └─> Sentence Transformers → Create vector embeddings        │
│     └─> Generate BOTH:                                          │
│         • Standard embedding (chunk only)                        │
│         • Contextual embedding (context + chunk)                │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                    TODO (Phase 2)                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  5. Vector Store (Qdrant)                                        │
│  6. BM25 Index (Lexical Search)                                  │
│  7. Hybrid Retrieval (Vector + BM25)                             │
│  8. Reranking                                                    │
│  9. End-to-End Pipeline                                          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## 📊 Data Flow

```
INPUT: document.pdf
         │
         ▼
    ┌────────────────┐
    │ Document Loader│
    └────────┬───────┘
             │ Full text
             ▼
    ┌────────────────┐
    │    Chunker     │
    └────────┬───────┘
             │ List of chunks
             │ [{chunk_text, chunk_id, ...}, ...]
             ▼
    ┌────────────────┐
    │ Contextualizer │◄─── Claude API (Haiku)
    └────────┬───────┘
             │ Chunks with context
             │ [{chunk_text, context, ...}, ...]
             ▼
    ┌────────────────┐
    │    Embedder    │◄─── Sentence Transformer
    └────────┬───────┘
             │ Enriched chunks
             │ [{chunk_text, context,
             │   embedding, contextual_embedding}, ...]
             ▼
         OUTPUT
```

## 📁 Project Structure

```
docagentContextual/
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── .env                      # API keys (not in git)
├── .env.example              # Example environment file
│
├── src/                      # Source code
│   ├── __init__.py
│   ├── document_loader.py    # ✅ PDF text extraction
│   ├── chunker.py            # ✅ Token-based chunking
│   ├── contextualizer.py     # ✅ Claude API integration
│   ├── embedder.py           # ✅ Vector embeddings
│   ├── vector_store.py       # ⏳ TODO: Qdrant integration
│   ├── bm25_index.py         # ⏳ TODO: Lexical search
│   ├── retriever.py          # ⏳ TODO: Hybrid retrieval
│   └── reranker.py           # ⏳ TODO: Result reranking
│
├── tests/                    # Test scripts
│   ├── __init__.py
│   ├── test_chunker.py       # Test chunking logic
│   ├── test_contextualizer.py# Test Claude API
│   └── test_embedder.py      # Test embeddings
│
└── data/                     # Document storage
    └── reference.docx        # Sample document
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd docagentContextual

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 2. Configuration

Edit `config.py` or set environment variables:

```python
# API Configuration
ANTHROPIC_API_KEY = "your-key-here"

# Chunking Settings
chunk_size = 800          # Tokens per chunk
chunk_overlap = 200       # Overlap between chunks

# Embedding Model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Claude Model
CLAUDE_MODEL = "claude-3-5-haiku-20241022"
```

### 3. Run Tests

```bash
# Test individual components
python tests/test_chunker.py
python tests/test_contextualizer.py
python tests/test_embedder.py
```

## 💡 Usage Example

```python
from src.document_loader import load_document
from src.chunker import chunk_text
from src.contextualizer import add_context_to_chunk
from src.embedder import Embedder

# 1. Load document
text = load_document("data/mydocument.pdf")

# 2. Create chunks
chunks = chunk_text(text, chunk_size_tokens=800, chunk_overlap=200)

# 3. Add context to each chunk
for chunk in chunks:
    chunk = add_context_to_chunk(chunk, text)

# 4. Generate embeddings
embedder = Embedder()
enriched_chunks = embedder.embed_chunks(chunks)

# Each chunk now has:
# - chunk_text: Original text
# - context: Contextual description
# - embedding: Standard embedding
# - contextual_embedding: Context + text embedding
```

## 📈 Performance Improvements

Based on Anthropic's research:

| Technique | Improvement |
|-----------|------------|
| Contextual Embeddings | **+35%** |
| Contextual BM25 | **+49%** (combined) |
| With Reranking | **+67%** (total) |

## 🔧 Technologies Used

- **Python 3.10+**
- **Anthropic Claude API** - Context generation (Haiku model)
- **Sentence Transformers** - Text embeddings (all-MiniLM-L6-v2)
- **tiktoken** - Token counting
- **PyPDF2** - PDF parsing
- **Qdrant** - Vector database (coming soon)
- **rank-bm25** - Lexical search (coming soon)

## 📝 Module Descriptions

### ✅ Implemented

#### 1. Document Loader (`src/document_loader.py`)
- Loads PDF documents
- Extracts text content
- Handles multiple document formats

#### 2. Chunker (`src/chunker.py`)
- Splits documents into manageable chunks
- Uses token-based chunking (not character-based)
- Configurable chunk size and overlap
- Preserves context with overlapping chunks

#### 3. Contextualizer (`src/contextualizer.py`) ⭐
- **Core innovation of the system**
- Uses Claude API to generate contextual descriptions
- Adds situational context to each chunk
- Example: "This chunk discusses Q3 revenue in ACME Corp's financial report"

#### 4. Embedder (`src/embedder.py`)
- Converts text to vector embeddings
- Uses Sentence Transformers (all-MiniLM-L6-v2)
- Generates two embeddings per chunk:
  - Standard embedding (baseline)
  - Contextual embedding (with added context)

### ⏳ TODO (Phase 2)

#### 5. Vector Store (`src/vector_store.py`)
- Qdrant integration
- Store and index embeddings
- Vector similarity search

#### 6. BM25 Index (`src/bm25_index.py`)
- Lexical search index
- Keyword-based retrieval
- Complement to vector search

#### 7. Hybrid Retriever (`src/retriever.py`)
- Combine vector + BM25 results
- Merge and deduplicate candidates

#### 8. Reranker (`src/reranker.py`)
- Score and rank final results
- Return top-N most relevant chunks

## 🎓 Learning Resources

- [Anthropic's Contextual Retrieval Blog Post](https://anthropic.com/news/contextual-retrieval)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Qdrant Vector Database](https://qdrant.tech/)

## 🤝 Contributing

This is a learning project! Feel free to:
- Add new features
- Improve existing code
- Add tests
- Enhance documentation

## 📄 License

MIT License - Feel free to use for learning and development

## 🙏 Acknowledgments

- Anthropic for the Contextual Retrieval technique
- Sentence Transformers team for the embedding models
- OpenAI for tiktoken tokenization

---

**Status**: Phase 1 Complete (4/9 modules) ✅  
**Next Up**: Qdrant Vector Store Integration
