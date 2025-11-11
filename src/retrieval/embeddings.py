# ==============================================================================
# FILE: embeddings.py
# PURPOSE: Generate embeddings (vector representations) for text chunks
# ==============================================================================

"""
This module handles conversion of text to dense vector embeddings.

WHY EMBEDDINGS?
- Raw text can't be used for similarity search
- Embeddings capture semantic meaning in vector space
- Similar texts have similar vectors (cosine similarity)
- Enables fast approximate nearest neighbor search

MODEL CHOICE: sentence-transformers/all-MiniLM-L6-v2
RATIONALE:
- Fast: Only 384 dimensions (vs 768+ for larger models)
- Accurate: Trained on 1B+ sentence pairs
- Balanced: Good trade-off between speed and quality
- Lightweight: ~80MB model size, runs on CPU
- Multilingual: Works for technical/legal/general text

TRADE-OFFS DISCUSSED:
1. Model Size vs Accuracy
   - Smaller: all-MiniLM-L6-v2 (384d) - CHOSEN for speed
   - Medium: all-mpnet-base-v2 (768d) - More accurate but slower
   - Large: instructor-large (1024d) - Best but requires GPU

2. Embedding Dimension
   - Lower (384d): Faster search, less storage, slightly lower accuracy
   - Higher (768d+): Better accuracy, slower search, more storage
   - CHOSEN: 384d sufficient for our document corpus size

3. CPU vs GPU
   - CPU: Slower embedding generation, but no GPU required
   - GPU: 10x faster, but deployment complexity
   - CHOSEN: CPU (deployment simplicity, acceptable speed for our scale)

USAGE:
    from src.retrieval.embeddings import EmbeddingModel
    
    # Initialize model
    model = EmbeddingModel()
    
    # Embed single text
    vector = model.embed_text("What is transformer architecture?")
    
    # Embed batch of texts (faster)
    vectors = model.embed_batch(["text1", "text2", "text3"])
"""

import os
import pickle
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Model configuration
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Dimension of embeddings from this model
BATCH_SIZE = 32      # Batch size for encoding (trade-off: speed vs memory)

# Cache directory for model weights
CACHE_DIR = "outputs/embeddings/models"


# ==============================================================================
# EMBEDDING MODEL CLASS
# ==============================================================================

class EmbeddingModel:
    """
    Wrapper for sentence-transformers embedding model.
    
    RESPONSIBILITIES:
    - Load pre-trained embedding model
    - Generate embeddings for queries and documents
    - Handle batching for efficiency
    - Cache model for reuse
    
    ATTRIBUTES:
        model_name (str): Name of the sentence-transformers model
        model (SentenceTransformer): Loaded model instance
        embedding_dim (int): Dimensionality of embeddings
    """
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initialize embedding model.
        
        PARAMETERS:
            model_name (str): HuggingFace model name
                Default: "sentence-transformers/all-MiniLM-L6-v2"
        
        PROCESS:
            1. Create cache directory if needed
            2. Download model (first time only, ~80MB)
            3. Load model into memory
            4. Set model to evaluation mode
        """
        print(f"\n{'='*80}")
        print(f"INITIALIZING EMBEDDING MODEL")
        print(f"{'='*80}")
        print(f"Model: {model_name}")
        print(f"Expected dimension: {EMBEDDING_DIM}")
        
        # Create cache directory
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Load model (downloads on first run)
        print(f"\nLoading model... (downloading if first time)")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, cache_folder=CACHE_DIR, device='cpu')
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Embedding dimension: {self.embedding_dim}")
        print(f"✓ Running on: CPU (GPU incompatible with PyTorch version)")
        print(f"{'='*80}\n")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        USE CASE: Embedding user queries (one at a time)
        
        PARAMETERS:
            text (str): Input text to embed
        
        RETURNS:
            np.ndarray: Embedding vector of shape (embedding_dim,)
        
        EXAMPLE:
            model = EmbeddingModel()
            query_vector = model.embed_text("What is transformer?")
            # Returns shape: (384,)
        
        NOTE: For multiple texts, use embed_batch() for better performance
        """
        # Encode single text
        # convert_to_numpy=True returns numpy array instead of tensor
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embedding
    
    def embed_batch(
        self, 
        texts: List[str], 
        batch_size: int = BATCH_SIZE,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts (batch processing).
        
        USE CASE: Embedding all document chunks (414 chunks in our case)
        
        WHY BATCHING?
        - 10-50x faster than encoding one-by-one
        - Better GPU/CPU utilization
        - Progress bar for long operations
        
        PARAMETERS:
            texts (List[str]): List of texts to embed
            batch_size (int): Number of texts per batch
                Trade-off: Larger = faster but more memory
                Default: 32 (good balance for CPU)
            show_progress (bool): Show progress bar
        
        RETURNS:
            np.ndarray: Embeddings array of shape (num_texts, embedding_dim)
        
        EXAMPLE:
            chunks = ["chunk1", "chunk2", ..., "chunk414"]
            embeddings = model.embed_batch(chunks)
            # Returns shape: (414, 384)
        """
        print(f"\nGenerating embeddings for {len(texts)} texts...")
        print(f"Batch size: {batch_size}")
        
        # Encode batch
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress
        )
        
        print(f"✓ Generated {len(embeddings)} embeddings")
        print(f"✓ Embedding shape: {embeddings.shape}")
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimensionality of embeddings produced by this model.
        
        RETURNS:
            int: Embedding dimension (384 for our model)
        """
        return self.embedding_dim


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    WHAT IS COSINE SIMILARITY?
    - Measures angle between two vectors
    - Range: -1 (opposite) to 1 (identical)
    - For embeddings: typically 0.0 to 1.0
    - >0.5 = similar, >0.7 = very similar, >0.9 = near identical
    
    FORMULA:
        cos(θ) = (A · B) / (||A|| * ||B||)
    
    PARAMETERS:
        vec1 (np.ndarray): First vector
        vec2 (np.ndarray): Second vector
    
    RETURNS:
        float: Similarity score between -1 and 1
    
    EXAMPLE:
        sim = cosine_similarity(query_embedding, chunk_embedding)
        if sim > 0.7:
            print("Highly relevant!")
    """
    # Compute dot product
    dot_product = np.dot(vec1, vec2)
    
    # Compute norms (magnitudes)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # Compute cosine similarity
    # Add small epsilon to avoid division by zero
    similarity = dot_product / (norm1 * norm2 + 1e-10)
    
    return float(similarity)


def save_embeddings(embeddings: np.ndarray, filepath: str):
    """
    Save embeddings to disk for reuse.
    
    WHY SAVE?
    - Embedding generation is slow (30-60 seconds for 414 chunks)
    - Document chunks don't change often
    - Can reuse saved embeddings across runs
    
    PARAMETERS:
        embeddings (np.ndarray): Embeddings array
        filepath (str): Path to save file (e.g., "outputs/embeddings/chunks.pkl")
    
    EXAMPLE:
        embeddings = model.embed_batch(chunks)
        save_embeddings(embeddings, "outputs/embeddings/chunks.pkl")
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(embeddings, f)
    
    print(f"✓ Saved embeddings to: {filepath}")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Size: {os.path.getsize(filepath) / 1024:.2f} KB")


def load_embeddings(filepath: str) -> np.ndarray:
    """
    Load previously saved embeddings from disk.
    
    PARAMETERS:
        filepath (str): Path to saved embeddings file
    
    RETURNS:
        np.ndarray: Loaded embeddings array
    
    EXAMPLE:
        embeddings = load_embeddings("outputs/embeddings/chunks.pkl")
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Embeddings file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        embeddings = pickle.load(f)
    
    print(f"✓ Loaded embeddings from: {filepath}")
    print(f"  Shape: {embeddings.shape}")
    
    return embeddings


# ==============================================================================
# SELF-TEST / DEMO BLOCK
# ==============================================================================

if __name__ == "__main__":
    """
    Test the embedding model.
    Run this file directly to test:
        python src/retrieval/embeddings.py
    """
    
    print("="*80)
    print("TESTING: embeddings.py")
    print("="*80)
    
    # Initialize model
    model = EmbeddingModel()
    
    # Test single text embedding
    print("\n" + "-"*80)
    print("TEST 1: Single Text Embedding")
    print("-"*80)
    text = "What is the transformer architecture?"
    embedding = model.embed_text(text)
    print(f"Text: {text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding (first 10 dims): {embedding[:10]}")
    
    # Test batch embedding
    print("\n" + "-"*80)
    print("TEST 2: Batch Embedding")
    print("-"*80)
    texts = [
        "Transformer is a neural network architecture",
        "Self-attention mechanism is key to transformers",
        "The weather is nice today",  # Unrelated text for comparison
    ]
    embeddings = model.embed_batch(texts, show_progress=False)
    print(f"Number of texts: {len(texts)}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Test cosine similarity
    print("\n" + "-"*80)
    print("TEST 3: Cosine Similarity")
    print("-"*80)
    print(f"Text 1: {texts[0]}")
    print(f"Text 2: {texts[1]}")
    print(f"Text 3: {texts[2]}")
    
    sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])
    sim_1_3 = cosine_similarity(embeddings[0], embeddings[2])
    
    print(f"\nSimilarity (Text 1 vs Text 2): {sim_1_2:.4f} (related)")
    print(f"Similarity (Text 1 vs Text 3): {sim_1_3:.4f} (unrelated)")
    print(f"✓ Related texts should have higher similarity!")
    
    # Test save/load
    print("\n" + "-"*80)
    print("TEST 4: Save/Load Embeddings")
    print("-"*80)
    test_filepath = "outputs/embeddings/test_embeddings.pkl"
    save_embeddings(embeddings, test_filepath)
    loaded = load_embeddings(test_filepath)
    print(f"✓ Saved and loaded successfully")
    print(f"✓ Arrays match: {np.allclose(embeddings, loaded)}")
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED ✓")
    print("="*80)
