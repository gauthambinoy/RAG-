# Preprocessing Package

"""
Preprocessing module for RAG system.

INCLUDES:
- Text normalization
- Smart chunking (document-aware)
- Chunk statistics
"""

from .chunker import normalize_text, chunk_text
from .smart_chunker import SmartChunker, chunk_documents, DocumentType
from .chunk_statistics import ChunkStatistics

__all__ = [
    'normalize_text',
    'chunk_text',
    'SmartChunker',
    'chunk_documents',
    'DocumentType',
    'ChunkStatistics'
]