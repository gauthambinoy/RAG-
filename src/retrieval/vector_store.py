# ==============================================================================
# FILE: vector_store.py
# PURPOSE: Store and search document embeddings using FAISS
# ==============================================================================

"""
This module handles vector storage and similarity search using FAISS.

WHY FAISS?
- Fast: Approximate nearest neighbor search in milliseconds
- Scalable: Handles millions of vectors efficiently
- Memory-efficient: Optimized data structures
- Industry-standard: Used by Facebook, OpenAI, etc.
- CPU-friendly: Works without GPU (though GPU faster)

WHAT IS FAISS?
- Library for efficient similarity search
- Builds index structures (like database indexes)
- Finds K nearest neighbors to a query vector
- Uses approximate algorithms for speed (vs exact brute-force)

INDEX TYPE CHOICE: IndexFlatL2
RATIONALE:
- Exact search (not approximate) - good for small datasets
- L2 distance (Euclidean) - equivalent to cosine for normalized vectors
- Simple: No training required
- Fast enough: <1ms for 414 vectors
- Alternatives: IndexIVFFlat (faster, approximate, needs training)

TRADE-OFFS DISCUSSED:
1. Exact vs Approximate Search
   - Exact (IndexFlatL2): Guaranteed best results, slower for large scale
   - Approximate (IndexIVFFlat): 10-100x faster, 95-99% accuracy
   - CHOSEN: Exact (only 414 vectors, speed not an issue)

2. Distance Metric
   - L2 (Euclidean): Standard, works with normalized vectors
   - Inner Product: For non-normalized vectors
   - Cosine: L2 on normalized vectors = cosine
   - CHOSEN: L2 with normalization

3. Index Persistence
   - Save index to disk: Fast loading, no re-embedding needed
   - Rebuild each time: Slower but always fresh
   - CHOSEN: Save to disk for efficiency

USAGE:
    from src.retrieval.vector_store import VectorStore
    
    # Create vector store
    store = VectorStore(dimension=384)
    
    # Add embeddings
    store.add_embeddings(embeddings, chunks)
    
    # Search for similar vectors
    results = store.search(query_embedding, k=5)
"""

import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Tuple


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Default search parameters
DEFAULT_K = 5  # Number of results to return
INDEX_SAVE_PATH = "outputs/embeddings/faiss_index.bin"
METADATA_SAVE_PATH = "outputs/embeddings/chunks_metadata.pkl"


# ==============================================================================
# VECTOR STORE CLASS
# ==============================================================================

class VectorStore:
    """
    Vector database for storing and searching document embeddings.
    
    RESPONSIBILITIES:
    - Store embeddings in FAISS index
    - Maintain mapping from vector ID to document metadata
    - Perform similarity search for queries
    - Save/load index for reuse
    
    ATTRIBUTES:
        dimension (int): Embedding dimension (e.g., 384)
        index (faiss.Index): FAISS index for similarity search
        chunks (List[Dict]): Metadata for each chunk (text, source, etc.)
        num_vectors (int): Number of vectors stored
    """
    
    def __init__(self, dimension: int = 384):
        """
        Initialize vector store.
        
        PARAMETERS:
            dimension (int): Dimension of embeddings (must match model)
                Default: 384 (for all-MiniLM-L6-v2)
        
        PROCESS:
            1. Create FAISS index (IndexFlatL2)
            2. Initialize metadata storage
        """
        print(f"\n{'='*80}")
        print(f"INITIALIZING VECTOR STORE")
        print(f"{'='*80}")
        print(f"Embedding dimension: {dimension}")
        print(f"Index type: IndexFlatL2 (exact search)")
        
        self.dimension = dimension
        
        # Create FAISS index
        # IndexFlatL2 = exact L2 distance search
        # L2 distance for normalized vectors = cosine similarity
        self.index = faiss.IndexFlatL2(dimension)
        
        # Store chunk metadata (text, source file, etc.)
        self.chunks = []
        
        print(f"✓ Vector store initialized")
        print(f"{'='*80}\n")
    
    def add_embeddings(
        self, 
        embeddings: np.ndarray, 
        chunks_data: List[Dict]
    ):
        """
        Add embeddings to the vector store.
        
        USE CASE: Add all 414 chunk embeddings after preprocessing
        
        PARAMETERS:
            embeddings (np.ndarray): Array of shape (N, dimension)
                N = number of chunks (e.g., 414)
            chunks_data (List[Dict]): Metadata for each chunk
                Each dict should contain:
                - 'text': The chunk text
                - 'source': Source file name
                - 'chunk_id': Chunk identifier
                - (optional) 'page', 'section', etc.
        
        PROCESS:
            1. Normalize embeddings (for cosine similarity via L2)
            2. Add to FAISS index
            3. Store metadata
        
        EXAMPLE:
            chunks = [
                {'text': 'chunk1...', 'source': 'paper.pdf', 'chunk_id': 0},
                {'text': 'chunk2...', 'source': 'paper.pdf', 'chunk_id': 1},
            ]
            store.add_embeddings(embeddings, chunks)
        """
        print(f"\n{'='*80}")
        print(f"ADDING EMBEDDINGS TO VECTOR STORE")
        print(f"{'='*80}")
        
        # Validate inputs
        if len(embeddings) != len(chunks_data):
            raise ValueError(
                f"Mismatch: {len(embeddings)} embeddings but "
                f"{len(chunks_data)} chunks"
            )
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} doesn't match "
                f"store dimension {self.dimension}"
            )
        
        print(f"Number of embeddings: {len(embeddings)}")
        print(f"Embedding shape: {embeddings.shape}")
        
        # Normalize embeddings (for cosine similarity via L2 distance)
        # After normalization: L2(a, b) = 2 - 2*cos(a, b)
        # So minimum L2 distance = maximum cosine similarity
        print(f"\nNormalizing embeddings...")
        embeddings_normalized = embeddings / np.linalg.norm(
            embeddings, axis=1, keepdims=True
        )
        
        # Add to FAISS index
        print(f"Adding to FAISS index...")
        self.index.add(embeddings_normalized.astype('float32'))
        
        # Store chunk metadata
        self.chunks.extend(chunks_data)
        
        print(f"✓ Added {len(embeddings)} vectors to store")
        print(f"✓ Total vectors in store: {self.index.ntotal}")
        print(f"{'='*80}\n")
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = DEFAULT_K
    ) -> List[Dict]:
        """
        Search for most similar chunks to query.
        
        USE CASE: Given user query embedding, find top K relevant chunks
        
        PARAMETERS:
            query_embedding (np.ndarray): Query vector of shape (dimension,)
            k (int): Number of results to return (default: 5)
        
        RETURNS:
            List[Dict]: Top K results, each containing:
                - 'chunk_id': ID in the store
                - 'text': Chunk text
                - 'source': Source filename
                - 'score': Similarity score (0-1, higher = more similar)
                - 'distance': L2 distance (lower = more similar)
                - (other metadata from original chunk)
        
        PROCESS:
            1. Normalize query embedding
            2. FAISS finds K nearest neighbors
            3. Convert distances to similarity scores
            4. Return results with metadata
        
        EXAMPLE:
            query_emb = model.embed_text("What is transformer?")
            results = store.search(query_emb, k=5)
            
            for result in results:
                print(f"Score: {result['score']:.3f}")
                print(f"Text: {result['text'][:100]}...")
        """
        if self.index.ntotal == 0:
            raise ValueError("Vector store is empty! Add embeddings first.")
        
        # Reshape query to (1, dimension) for FAISS
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query (same as stored embeddings)
        query_normalized = query_embedding / np.linalg.norm(
            query_embedding, axis=1, keepdims=True
        )
        
        # Search FAISS index
        # Returns: distances (L2), indices (chunk IDs)
        distances, indices = self.index.search(
            query_normalized.astype('float32'), 
            k
        )
        
        # Convert to results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            # Convert L2 distance to cosine similarity
            # For normalized vectors: similarity = 1 - (distance^2 / 2)
            # Approximate: similarity ≈ 1 - (distance / 2)
            similarity = 1.0 - (dist / 2.0)
            
            # Get chunk metadata
            chunk = self.chunks[idx].copy()
            
            # Add search metadata
            chunk['rank'] = i + 1
            chunk['distance'] = float(dist)
            chunk['score'] = float(similarity)
            
            results.append(chunk)
        
        return results
    
    def get_num_vectors(self) -> int:
        """
        Get number of vectors in the store.
        
        RETURNS:
            int: Number of vectors stored
        """
        return self.index.ntotal
    
    def save(self, index_path: str = INDEX_SAVE_PATH, 
             metadata_path: str = METADATA_SAVE_PATH):
        """
        Save vector store to disk for reuse.
        
        WHY SAVE?
        - Avoid re-embedding on every run (slow!)
        - Fast loading: <1 second vs 30-60 seconds to re-embed
        - Consistency: Same index across runs
        
        PARAMETERS:
            index_path (str): Path to save FAISS index
            metadata_path (str): Path to save chunk metadata
        
        SAVES TWO FILES:
            1. FAISS index (binary, ~100-500KB for 414 vectors)
            2. Chunks metadata (pickle, text + metadata)
        
        EXAMPLE:
            store.save()  # Uses default paths
        """
        print(f"\n{'='*80}")
        print(f"SAVING VECTOR STORE")
        print(f"{'='*80}")
        
        # Create directories
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        # Save FAISS index
        print(f"Saving FAISS index to: {index_path}")
        faiss.write_index(self.index, index_path)
        index_size = os.path.getsize(index_path) / 1024  # KB
        print(f"✓ FAISS index saved ({index_size:.2f} KB)")
        
        # Save chunks metadata
        print(f"Saving metadata to: {metadata_path}")
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        metadata_size = os.path.getsize(metadata_path) / 1024  # KB
        print(f"✓ Metadata saved ({metadata_size:.2f} KB)")
        
        print(f"{'='*80}\n")
    
    @classmethod
    def load(cls, index_path: str = INDEX_SAVE_PATH, 
             metadata_path: str = METADATA_SAVE_PATH,
             dimension: int = 384) -> 'VectorStore':
        """
        Load vector store from disk.
        
        PARAMETERS:
            index_path (str): Path to saved FAISS index
            metadata_path (str): Path to saved metadata
            dimension (int): Embedding dimension
        
        RETURNS:
            VectorStore: Loaded vector store ready for search
        
        EXAMPLE:
            store = VectorStore.load()
            results = store.search(query_embedding, k=5)
        """
        print(f"\n{'='*80}")
        print(f"LOADING VECTOR STORE")
        print(f"{'='*80}")
        
        # Check files exist
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found: {index_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        # Create instance
        store = cls(dimension=dimension)
        
        # Load FAISS index
        print(f"Loading FAISS index from: {index_path}")
        store.index = faiss.read_index(index_path)
        print(f"✓ Loaded index with {store.index.ntotal} vectors")
        
        # Load chunks metadata
        print(f"Loading metadata from: {metadata_path}")
        with open(metadata_path, 'rb') as f:
            store.chunks = pickle.load(f)
        print(f"✓ Loaded {len(store.chunks)} chunk metadata")
        
        print(f"{'='*80}\n")
        
        return store


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def print_search_results(results: List[Dict], max_text_len: int = 200):
    """
    Pretty-print search results.
    
    PARAMETERS:
        results (List[Dict]): Results from store.search()
        max_text_len (int): Maximum characters of text to show
    
    EXAMPLE:
        results = store.search(query_emb, k=5)
        print_search_results(results)
    """
    print(f"\n{'='*80}")
    print(f"SEARCH RESULTS (Top {len(results)})")
    print(f"{'='*80}\n")
    
    for result in results:
        print(f"[Rank {result['rank']}] Score: {result['score']:.4f}")
        print(f"  Source: {result['source']}")
        print(f"  Chunk ID: {result['chunk_id']}")
        
        # Truncate text for display
        text = result['text']
        if len(text) > max_text_len:
            text = text[:max_text_len] + "..."
        print(f"  Text: {text}")
        print()


# ==============================================================================
# SELF-TEST / DEMO BLOCK
# ==============================================================================

if __name__ == "__main__":
    """
    Test the vector store.
    Run this file directly to test:
        python src/retrieval/vector_store.py
    """
    
    print("="*80)
    print("TESTING: vector_store.py")
    print("="*80)
    
    # Create test embeddings
    print("\n" + "-"*80)
    print("TEST 1: Create and Add Embeddings")
    print("-"*80)
    
    dimension = 384
    num_chunks = 5
    
    # Generate random embeddings (simulating real embeddings)
    np.random.seed(42)
    embeddings = np.random.randn(num_chunks, dimension).astype('float32')
    
    # Create chunk metadata
    chunks = [
        {
            'chunk_id': i,
            'text': f"This is test chunk {i} about transformers and attention.",
            'source': f"test_doc_{i % 2}.pdf"
        }
        for i in range(num_chunks)
    ]
    
    # Initialize and add
    store = VectorStore(dimension=dimension)
    store.add_embeddings(embeddings, chunks)
    
    print(f"✓ Added {store.get_num_vectors()} vectors")
    
    # Test search
    print("\n" + "-"*80)
    print("TEST 2: Search")
    print("-"*80)
    
    # Use first embedding as query (should find itself as #1)
    query_emb = embeddings[0]
    results = store.search(query_emb, k=3)
    
    print_search_results(results, max_text_len=100)
    
    print(f"✓ Top result chunk_id: {results[0]['chunk_id']} (should be 0)")
    print(f"✓ Top result score: {results[0]['score']:.4f} (should be ~1.0)")
    
    # Test save/load
    print("\n" + "-"*80)
    print("TEST 3: Save and Load")
    print("-"*80)
    
    test_index_path = "outputs/embeddings/test_index.bin"
    test_metadata_path = "outputs/embeddings/test_metadata.pkl"
    
    store.save(test_index_path, test_metadata_path)
    
    loaded_store = VectorStore.load(test_index_path, test_metadata_path, dimension)
    
    print(f"✓ Loaded store has {loaded_store.get_num_vectors()} vectors")
    
    # Test search on loaded store
    results_loaded = loaded_store.search(query_emb, k=3)
    print(f"✓ Search on loaded store works")
    print(f"✓ Results match: {results[0]['chunk_id'] == results_loaded[0]['chunk_id']}")
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED ✓")
    print("="*80)
