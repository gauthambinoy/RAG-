# ==============================================================================
# FILE: preprocess.py
# PURPOSE: Text normalization and chunking for RAG pipeline
# ==============================================================================

"""
This module handles all preprocessing steps AFTER loading documents.

PREPROCESSING PIPELINE:
1. Load raw text from documents (done by load_*.py files)
2. Normalize text (this file) - clean, lowercase, remove noise
3. Chunk text (this file) - split into searchable pieces
4. Ready for embedding and indexing

WHY SEPARATE FROM LOADERS?
- Single Responsibility: Loaders extract, preprocessors clean
- Reusable: Same preprocessing for all document types
- Testable: Easy to verify each step independently
- Flexible: Can adjust preprocessing without touching loaders
"""

# --- Import Libraries ---
import re          # Regular expressions for text cleaning
import unicodedata # Unicode normalization for special characters

# ==============================================================================
# FUNCTION 1: Text Normalization
# ==============================================================================

def normalize_text(text, lowercase=True, remove_extra_spaces=True, 
                   remove_special_chars=False, normalize_unicode=True):
    """
    Clean and normalize text to improve RAG retrieval quality.
    
    WHY NORMALIZE?
    - Makes search more consistent (e.g., "AI" vs "ai" both match)
    - Removes noise that doesn't help with meaning
    - Standardizes text format across different document types
    
    PREPROCESSING DECISIONS & RATIONALE:
    
    1. LOWERCASE (Default: True)
       WHY: "Machine Learning" and "machine learning" should be treated the same
       TRADE-OFF: Loses proper nouns distinction, but improves recall
       
    2. UNICODE NORMALIZATION (Default: True)
       WHY: "café" (with é) vs "cafe" (with e+accent) should match
       USES: NFKD normalization - decomposes characters to base form
       EXAMPLE: "é" → "e" + accent mark, then remove accent
       
    3. EXTRA SPACES (Default: True)
       WHY: PDFs often have inconsistent spacing
       EXAMPLE: "Hello    world" → "Hello world"
       
    4. SPECIAL CHARACTERS (Default: False)
       WHY: Keep by default because they might be important
       EXAMPLE: "C++" vs "C" are different programming languages
       USE CASE: Set to True if you want only alphanumeric text
    
    PARAMETERS:
        text (str): Raw text from document
        lowercase (bool): Convert to lowercase
        remove_extra_spaces (bool): Collapse multiple spaces to single space
        remove_special_chars (bool): Remove non-alphanumeric characters
        normalize_unicode (bool): Normalize special Unicode characters
        
    RETURNS:
        str: Cleaned and normalized text
    """
    
    if not text or not isinstance(text, str):
        return ""
    
    # --- STEP 1: Unicode Normalization ---
    # DECISION: Handle special characters like accents, symbols, etc.
    # This is important for documents with international characters
    if normalize_unicode:
        # NFKD = Compatibility Decomposition
        # Breaks down complex characters into simpler forms
        # Example: "ﬁ" (single char) → "fi" (two chars)
        text = unicodedata.normalize('NFKD', text)
        
        # Remove accent marks and other combining characters
        # Keep only ASCII characters (basic English alphabet + numbers)
        text = ''.join([c for c in text if not unicodedata.combining(c)])
    
    # --- STEP 2: Lowercase Conversion ---
    # DECISION: Treat "AI" and "ai" as the same concept
    # Most RAG systems benefit from case-insensitive matching
    if lowercase:
        text = text.lower()
    
    # --- STEP 3: Remove Special Characters (Optional) ---
    # DECISION: Only do this if specifically requested
    # Default is FALSE because special chars can be meaningful
    # Example: Keep "C++" as is, don't make it "C"
    if remove_special_chars:
        # Keep only letters, numbers, and spaces
        # \w matches [a-zA-Z0-9_]
        text = re.sub(r'[^\w\s]', ' ', text)
    
    # --- STEP 4: Whitespace Normalization ---
    # DECISION: PDFs often have weird spacing from column layouts
    # Collapse all whitespace (spaces, tabs, newlines) into single spaces
    if remove_extra_spaces:
        # Replace one or more whitespace characters with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
    
    return text


# ==============================================================================
# FUNCTION 2: Text Chunking
# ==============================================================================

def chunk_text(text, chunk_size=800, overlap=100, preserve_sentences=True):
    """
    Split long text into smaller, overlapping chunks for RAG retrieval.
    
    WHY CHUNK?
    - Embedding models have token limits (e.g., 512 tokens)
    - Smaller chunks = more precise retrieval
    - Overlap ensures context isn't lost at boundaries
    
    CHUNKING DECISIONS & RATIONALE:
    
    1. CHUNK SIZE (Default: 800 characters)
       WHY 800?
       - Roughly 150-200 tokens (most embedding models support 512+)
       - Large enough to contain meaningful context
       - Small enough for precise retrieval
       ALTERNATIVE: Could use 500 (smaller) or 1200 (larger)
       TRADE-OFF: Smaller = more precise but less context
       
    2. OVERLAP (Default: 100 characters)
       WHY OVERLAP?
       - Prevents important info from being split at boundaries
       - If a sentence is cut in half, the overlap captures it
       - 100 chars ≈ 1-2 sentences of overlap
       EXAMPLE: 
         Chunk 1: "...end of sentence. Start of next sentence..."
         Chunk 2: "...Start of next sentence. More content..." (overlap)
       
    3. SENTENCE PRESERVATION (Default: True)
       WHY?
       - Don't cut in the middle of a sentence
       - Improves semantic coherence of chunks
       - Better for LLM context understanding
       HOW: Try to break at sentence boundaries (. ! ?)
    
    PARAMETERS:
        text (str): Normalized text to chunk
        chunk_size (int): Target size of each chunk in characters
        overlap (int): Number of overlapping characters between chunks
        preserve_sentences (bool): Try to break at sentence boundaries
        
    RETURNS:
        list[dict]: List of chunks with metadata
        Example: [
            {
                'text': 'chunk content...',
                'chunk_id': 0,
                'start_pos': 0,
                'end_pos': 800,
                'length': 800
            },
            ...
        ]
    """
    
    if not text or not isinstance(text, str):
        return []
    
    chunks = []
    text_length = len(text)
    
    # If text is shorter than chunk size, return as single chunk
    if text_length <= chunk_size:
        return [{
            'text': text,
            'chunk_id': 0,
            'start_pos': 0,
            'end_pos': text_length,
            'length': text_length
        }]
    
    # --- STEP 1: Create chunks with sliding window ---
    # SLIDING WINDOW APPROACH:
    # Start at position 0, move forward by (chunk_size - overlap) each time
    # This creates overlapping chunks
    
    chunk_id = 0
    start_pos = 0
    
    while start_pos < text_length:
        # Calculate end position for this chunk
        end_pos = start_pos + chunk_size
        
        # Don't go past the end of the text
        if end_pos > text_length:
            end_pos = text_length
        
        # Extract the chunk
        chunk_text = text[start_pos:end_pos]
        
        # --- STEP 2: Sentence Boundary Preservation ---
        # DECISION: If we're not at the end of the text, try to break at a sentence
        # This keeps chunks more coherent and meaningful
        if preserve_sentences and end_pos < text_length:
            # Look for sentence endings: . ! ?
            # Search in the last 20% of the chunk to find a good break point
            search_start = int(len(chunk_text) * 0.8)  # Last 20% of chunk
            last_part = chunk_text[search_start:]
            
            # Find the last sentence ending in this part
            # We look for: period, exclamation, or question mark followed by space
            sentence_endings = [
                last_part.rfind('. '),
                last_part.rfind('! '),
                last_part.rfind('? ')
            ]
            
            # Get the position of the last sentence ending
            last_sentence_end = max(sentence_endings)
            
            # If we found a sentence boundary, break there
            if last_sentence_end > 0:
                # Adjust chunk to end at sentence boundary
                actual_end = search_start + last_sentence_end + 1  # +1 to include the period
                chunk_text = chunk_text[:actual_end]
                end_pos = start_pos + actual_end
        
        # --- STEP 3: Store chunk with metadata ---
        # WHY METADATA?
        # - chunk_id: Track order of chunks
        # - start_pos/end_pos: Know where chunk came from in original doc
        # - length: Useful for debugging and analysis
        chunks.append({
            'text': chunk_text.strip(),  # Remove any trailing whitespace
            'chunk_id': chunk_id,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'length': len(chunk_text.strip())
        })
        
        # --- STEP 4: Move to next chunk position ---
        # DECISION: Move forward by (chunk_size - overlap)
        # This creates the overlap between consecutive chunks
        # EXAMPLE: chunk_size=800, overlap=100
        #   Chunk 1: chars 0-800
        #   Chunk 2: chars 700-1500 (100 char overlap)
        #   Chunk 3: chars 1400-2200 (100 char overlap)
        start_pos += (chunk_size - overlap)
        chunk_id += 1
    
    return chunks


# ==============================================================================
# FUNCTION 3: Full Preprocessing Pipeline
# ==============================================================================

def preprocess_document(text, normalize=True, chunk=True, 
                       chunk_size=800, overlap=100):
    """
    Complete preprocessing pipeline: normalize → chunk
    
    This is the main function you should use in your RAG pipeline.
    It combines normalization and chunking in the correct order.
    
    PIPELINE ORDER:
    1. Normalize text (clean, lowercase, etc.)
    2. Chunk normalized text (split into pieces)
    
    WHY THIS ORDER?
    - Normalize first so all chunks have consistent formatting
    - Chunk after so we're splitting clean text
    
    PARAMETERS:
        text (str): Raw text from document loader
        normalize (bool): Whether to normalize text
        chunk (bool): Whether to chunk text
        chunk_size (int): Target chunk size in characters
        overlap (int): Overlap between chunks in characters
        
    RETURNS:
        If chunk=True: list[dict] of chunks with metadata
        If chunk=False: str of normalized text
        
    EXAMPLE USAGE:
        # Load document
        raw_text = load_pdf_data("paper.pdf")
        
        # Preprocess it
        chunks = preprocess_document(raw_text, 
                                     normalize=True, 
                                     chunk=True,
                                     chunk_size=800,
                                     overlap=100)
        
        # Now chunks are ready for embedding!
        for chunk in chunks:
            print(f"Chunk {chunk['chunk_id']}: {chunk['text'][:100]}...")
    """
    
    if not text or not isinstance(text, str):
        return [] if chunk else ""
    
    # --- STEP 1: Text Normalization ---
    if normalize:
        text = normalize_text(
            text,
            lowercase=True,
            remove_extra_spaces=True,
            remove_special_chars=False,  # Keep special chars by default
            normalize_unicode=True
        )
    
    # --- STEP 2: Text Chunking ---
    if chunk:
        chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=overlap,
            preserve_sentences=True
        )
        return chunks
    else:
        # Return normalized text without chunking
        return text


# ==============================================================================
# FUNCTION 4: Display Preprocessing Results to User
# ==============================================================================

def show_preprocessing_results(original_text, processed_chunks, 
                               document_name="Document",
                               show_original_sample=True,
                               show_normalized_sample=True,
                               show_chunk_samples=True,
                               max_sample_length=400,
                               max_chunks_to_show=3):
    """
    Display preprocessing results in a user-friendly format.
    
    WHY SHOW THIS TO USERS?
    - Transparency: Users see what we did to their data
    - Validation: Users can verify preprocessing worked correctly
    - Documentation: Creates a record for challenge submission
    - Debugging: Easy to spot issues in preprocessing
    - Trust: Users understand the transformation process
    
    PARAMETERS:
        original_text (str): Raw text before preprocessing
        processed_chunks (list[dict]): Chunks after preprocessing
        document_name (str): Name of the document being processed
        show_original_sample (bool): Show sample of original text
        show_normalized_sample (bool): Show normalized version
        show_chunk_samples (bool): Show individual chunk examples
        max_sample_length (int): How many characters to show in samples
        max_chunks_to_show (int): Maximum number of chunks to display
        
    RETURNS:
        None (prints to console in user-friendly format)
        
    EXAMPLE USAGE:
        raw_text = load_pdf_data("paper.pdf")
        chunks = preprocess_document(raw_text)
        show_preprocessing_results(raw_text, chunks, "paper.pdf")
    """
    
    print("\n" + "="*80)
    print(f"📊 PREPROCESSING RESULTS: {document_name}")
    print("="*80)
    
    # --- SECTION 1: Show Original Text Sample ---
    if show_original_sample and original_text:
        print("\n📄 STEP 1: ORIGINAL TEXT (Before Preprocessing)")
        print("-" * 80)
        sample = original_text[:max_sample_length]
        print(sample)
        if len(original_text) > max_sample_length:
            print(f"\n... (showing first {max_sample_length} of {len(original_text):,} characters)")
        else:
            print(f"\n(Complete text shown: {len(original_text):,} characters)")
    
    # --- SECTION 2: Show What Normalization Did ---
    if show_normalized_sample and processed_chunks:
        print("\n\n🔧 STEP 2: NORMALIZED TEXT (After Cleaning)")
        print("-" * 80)
        
        # Get text from first chunk as sample of normalized text
        if len(processed_chunks) > 0:
            first_chunk_text = processed_chunks[0]['text']
            sample = first_chunk_text[:max_sample_length]
            print(sample)
            if len(first_chunk_text) > max_sample_length:
                print(f"\n... (showing first {max_sample_length} characters)")
        
        print("\n✨ NORMALIZATION CHANGES APPLIED:")
        print("   ✓ Converted to lowercase (e.g., 'AI' → 'ai')")
        print("   ✓ Removed extra whitespace (multiple spaces → single space)")
        print("   ✓ Normalized Unicode characters (é → e)")
        print("   ✓ Preserved special characters (kept C++, punctuation, etc.)")
    
    # --- SECTION 3: Show Chunking Statistics ---
    if processed_chunks:
        print("\n\n📦 STEP 3: CHUNKING STATISTICS")
        print("-" * 80)
        
        total_chunks = len(processed_chunks)
        total_chars = sum(chunk['length'] for chunk in processed_chunks)
        avg_chunk_size = total_chars / total_chunks if total_chunks > 0 else 0
        
        print(f"   📊 Total Chunks Created: {total_chunks}")
        print(f"   📏 Total Characters: {total_chars:,}")
        print(f"   📐 Average Chunk Size: {avg_chunk_size:.0f} characters (~{avg_chunk_size/5:.0f} tokens)")
        print(f"   🔗 Overlap Between Chunks: 100 characters")
        
        # Show size distribution
        if total_chunks > 0:
            min_size = min(chunk['length'] for chunk in processed_chunks)
            max_size = max(chunk['length'] for chunk in processed_chunks)
            print(f"   📊 Chunk Size Range: {min_size} - {max_size} characters")
        
        print("\n   💡 WHY CHUNKING?")
        print("      - Embedding models have token limits (typically 512 tokens)")
        print("      - Smaller chunks = more precise retrieval")
        print("      - Overlap ensures no information is lost at boundaries")
    
    # --- SECTION 4: Show Individual Chunk Samples ---
    if show_chunk_samples and processed_chunks:
        print("\n\n📝 STEP 4: CHUNK SAMPLES")
        print("-" * 80)
        
        chunks_to_display = min(max_chunks_to_show, len(processed_chunks))
        print(f"Showing {chunks_to_display} of {len(processed_chunks)} chunks:\n")
        
        for i in range(chunks_to_display):
            chunk = processed_chunks[i]
            print(f"┌─ Chunk #{chunk['chunk_id']} " + "─" * 63)
            print(f"│ 📍 Position: characters {chunk['start_pos']:,} - {chunk['end_pos']:,}")
            print(f"│ 📏 Length: {chunk['length']} characters (~{chunk['length']/5:.0f} tokens)")
            print(f"│")
            
            # Show chunk text (truncated if too long)
            chunk_text = chunk['text']
            if len(chunk_text) > 250:
                print(f"│ 📄 Text Preview:")
                print(f"│    {chunk_text[:250]}...")
            else:
                print(f"│ 📄 Text:")
                print(f"│    {chunk_text}")
            print(f"└{'─' * 78}\n")
        
        if len(processed_chunks) > max_chunks_to_show:
            remaining = len(processed_chunks) - max_chunks_to_show
            print(f"   ... and {remaining} more chunks (not shown)")
    
    # --- SECTION 5: Show Readiness Status ---
    print("\n" + "="*80)
    print("✅ PREPROCESSING COMPLETE - DATA IS READY!")
    print("="*80)
    print("\n📌 Your preprocessed data is now ready for:")
    print("   1️⃣  Embedding Generation (convert text to vectors)")
    print("   2️⃣  Vector Store Indexing (store in FAISS/ChromaDB)")
    print("   3️⃣  Semantic Search (find relevant chunks)")
    print("   4️⃣  RAG Pipeline (retrieval + generation)")
    print("\n" + "="*80 + "\n")


# ==============================================================================
# SELF-TEST / DEMO BLOCK
# ==============================================================================

if __name__ == "__main__":
    """
    Test the preprocessing functions with example text.
    Run this file directly to see preprocessing in action:
        python preprocess.py
    """
    
    print("="*80)
    print("TESTING: preprocess.py")
    print("="*80)
    
    # --- Test Text ---
    # Deliberately messy text to show what preprocessing does
    test_text = """
    The Transformer   architecture was introduced in the paper 
    "Attention Is All You Need" by Vaswani et al. (2017).    
    It revolutionized NLP!  The model uses self-attention mechanisms.
    
    This allows it to process    sequences in parallel, unlike RNNs.
    The architecture consists of an encoder and a decoder.    Each has 
    multiple layers.   
    
    Key innovations include: multi-head attention, positional encodings,
    and layer normalization. These components work together to create 
    powerful language models like GPT and BERT.
    """
    
    print("\nRunning preprocessing on sample text...")
    print("This demonstrates the complete pipeline:\n")
    
    # --- Run Full Preprocessing Pipeline ---
    processed_chunks = preprocess_document(
        test_text,
        normalize=True,
        chunk=True,
        chunk_size=150,
        overlap=30
    )
    
    # --- Display Results to User ---
    show_preprocessing_results(
        original_text=test_text,
        processed_chunks=processed_chunks,
        document_name="Sample Transformer Paper Text",
        show_original_sample=True,
        show_normalized_sample=True,
        show_chunk_samples=True,
        max_sample_length=300,
        max_chunks_to_show=3
    )
    
    print("="*80)
    print("✅ TEST COMPLETE - Preprocessing module is working correctly!")
    print("="*80)