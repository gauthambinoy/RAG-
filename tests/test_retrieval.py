# ==============================================================================
# FILE: test_retrieval.py
# PURPOSE: Test retrieval component with real preprocessed documents
# ==============================================================================

"""
Test the complete retrieval pipeline with actual documents.

WHAT THIS TESTS:
1. Load preprocessed documents (414 chunks)
2. Build retrieval index with embeddings
3. Test queries from test_queries.py
4. Evaluate retrieval quality

RUN:
    python tests/test_retrieval.py
"""

import sys
import os

# Add src and tests to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.test_pipeline import find_and_load_all_documents, preprocess_all_documents
from src.retrieval.retriever import Retriever
from src.queries.test_queries import get_all_queries, get_query_by_id


# ==============================================================================
# MAIN TEST
# ==============================================================================

def test_retrieval_pipeline():
    """
    Test complete retrieval pipeline with real documents.
    """
    
    print("="*80)
    print("TESTING RETRIEVAL PIPELINE")
    print("="*80)
    
    # Step 1: Load preprocessed documents
    print("\n" + "="*80)
    print("STEP 1: LOAD PREPROCESSED DOCUMENTS")
    print("="*80)
    
    # Load raw documents first
    loaded_documents = find_and_load_all_documents()
    
    # Preprocess them into chunks (returns dict: filename -> chunks)
    chunks_by_file = preprocess_all_documents(loaded_documents, chunk_size=800, overlap=100)
    
    # Flatten to single list of chunks with proper metadata
    chunks = []
    for filename, file_chunks in chunks_by_file.items():
        for chunk in file_chunks:
            # Ensure each chunk has required fields for retrieval
            chunk_data = {
                'text': chunk['text'],
                'source': filename,
                'chunk_id': len(chunks),  # Unique ID across all chunks
                'file_chunk_id': chunk.get('chunk_id', 0)  # ID within file
            }
            chunks.append(chunk_data)
    
    print(f"\n✓ Loaded {len(chunks)} document chunks")
    
    # Step 2: Build retrieval index
    print("\n" + "="*80)
    print("STEP 2: BUILD RETRIEVAL INDEX")
    print("="*80)
    
    retriever = Retriever()
    retriever.build_index(chunks, use_cache=True, save_to_cache=True)
    
    print(f"\n✓ Index built with {retriever.vector_store.get_num_vectors()} vectors")
    
    # Step 3: Test with sample queries
    print("\n" + "="*80)
    print("STEP 3: TEST RETRIEVAL WITH SAMPLE QUERIES")
    print("="*80)
    
    # Test 3 queries of different difficulties
    test_query_ids = [1, 5, 9]  # Easy, medium, hard
    
    for query_id in test_query_ids:
        query_obj = get_query_by_id(query_id)
        
        print(f"\n{'─'*80}")
        print(f"QUERY {query_id}: {query_obj['query']}")
        print(f"Category: {query_obj['category']} | Difficulty: {query_obj['difficulty']}")
        print(f"Expected sources: {', '.join(query_obj['expected_sources'])}")
        print(f"{'─'*80}")
        
        # Retrieve
        results = retriever.retrieve(
            query_obj['query'], 
            k=5,
            min_score=0.3,
            verbose=False
        )
        
        print(f"\nRetrieved {len(results)} relevant chunks:\n")
        
        for i, result in enumerate(results, 1):
            print(f"[{i}] Score: {result['score']:.3f} | Source: {result['source']}")
            print(f"    Text preview: {result['text'][:150]}...")
            print()
        
        # Check if expected sources are in results
        retrieved_sources = set(r['source'] for r in results)
        expected_sources = set(query_obj['expected_sources'])
        
        match = retrieved_sources & expected_sources
        if match:
            print(f"✓ SUCCESS: Found expected sources: {', '.join(match)}")
        else:
            print(f"⚠ WARNING: Expected sources not in top results")
            print(f"  Expected: {', '.join(expected_sources)}")
            print(f"  Retrieved: {', '.join(retrieved_sources)}")
    
    # Step 4: Test context formatting for LLM
    print("\n" + "="*80)
    print("STEP 4: TEST CONTEXT FORMATTING FOR LLM")
    print("="*80)
    
    query = "What is the transformer architecture?"
    context = retriever.get_context_for_llm(query, k=3)
    
    print(f"\nQuery: {query}\n")
    print("Formatted context for LLM:")
    print("-" * 80)
    print(context)
    print("-" * 80)
    
    print("\n" + "="*80)
    print("✓ RETRIEVAL PIPELINE TEST COMPLETE")
    print("="*80)
    
    # Basic assertions to validate retrieval pipeline
    assert retriever.is_ready is True
    assert retriever.vector_store.get_num_vectors() == len(chunks)

    # Ensure context formatting returns non-empty string
    assert isinstance(context, str) and len(context) > 0


# ==============================================================================
# RUN TESTS
# ==============================================================================

if __name__ == "__main__":
    try:
        test_retrieval_pipeline()
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
    except Exception as e:
        print(f"\n❌ TEST FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
