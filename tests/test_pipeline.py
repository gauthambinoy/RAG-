# ==============================================================================
# FILE: test.py
# PURPOSE: Test all data loaders and PREPROCESSING pipeline
# ==============================================================================

# --- Import Our Custom Loaders ---
# These are the 3 loader files we just created
from src.loaders.excel_loader import load_excel_or_csv  # For Excel and CSV files
from src.loaders.docx_loader import load_docx_data      # For Word documents (.docx)
from src.loaders.pdf_loader import load_pdf_data        # For PDF files

# --- Import Our Preprocessing Module ---
from src.preprocessing.normalizer import preprocess_document, show_preprocessing_results  # For text normalization and chunking

# --- Import Standard Libraries ---
import os        # For file path operations
import glob      # For finding files with patterns (*.pdf, *.docx, etc.)

# ==============================================================================
# MAIN FUNCTION: Find and load all documents
# ==============================================================================

def find_and_load_all_documents():
    """
    This is the main function that:
    1. Searches for ALL PDF, DOCX, and Excel files in the workspace
    2. Loads each file using the appropriate loader
    3. Stores all the content in a dictionary
    4. Prints a summary
    
    WHY? We need to test that all our loaders work and see what data we have
    for the RAG system.
    """
    
    print("="*80)
    print("STARTING: Document Discovery and Loading Test")
    print("="*80)
    
    # --- STEP 1: Define file patterns to search for ---
    # DECISION: We look for specific file types that our loaders can handle
    # We search in both the root directory AND the DATA/ subdirectory
    
    search_locations = [
        "data/pdfs",          # PDF documents folder
        "data/documents",     # DOCX documents folder  
        "data/tables",        # Excel/CSV tables folder
        "data/raw_documents"  # Fallback for raw documents (common in this repo)
    ]
    
    file_patterns = {
        'pdf': '*.pdf',      # All PDF files
        'docx': '*.docx',    # All Word documents
        'excel': '*.xlsx',   # Excel files (modern format)
        'excel_old': '*.xls' # Excel files (old format)
    }
    
    # --- STEP 2: Find all matching files ---
    print("\n" + "="*80)
    print("STEP 1: Discovering Files")
    print("="*80)
    
    all_files = {
        'pdf': [],
        'docx': [],
        'excel': []
    }
    
    # Search each location
    for location in search_locations:
        if not os.path.exists(location):
            continue
            
        print(f"\nSearching in: {location}/")
        
        # Find PDF files
        pdf_files = glob.glob(os.path.join(location, file_patterns['pdf']))
        for f in pdf_files:
            all_files['pdf'].append(f)
            print(f"  [PDF] Found: {f}")
        
        # Find DOCX files
        docx_files = glob.glob(os.path.join(location, file_patterns['docx']))
        for f in docx_files:
            all_files['docx'].append(f)
            print(f"  [DOCX] Found: {f}")
        
        # Find Excel files (both .xlsx and .xls)
        xlsx_files = glob.glob(os.path.join(location, file_patterns['excel']))
        xls_files = glob.glob(os.path.join(location, file_patterns['excel_old']))
        for f in xlsx_files + xls_files:
            all_files['excel'].append(f)
            print(f"  [EXCEL] Found: {f}")
    
    # --- STEP 3: Print summary of what we found ---
    print("\n" + "="*80)
    print("DISCOVERY SUMMARY")
    print("="*80)
    print(f"Total PDFs found: {len(all_files['pdf'])}")
    print(f"Total DOCX found: {len(all_files['docx'])}")
    print(f"Total Excel found: {len(all_files['excel'])}")
    print(f"GRAND TOTAL: {sum(len(v) for v in all_files.values())} files")
    
    # --- STEP 4: Load each file with the appropriate loader ---
    print("\n" + "="*80)
    print("STEP 2: Loading All Documents")
    print("="*80)
    
    # Dictionary to store all loaded content
    # Key = filename, Value = text content
    loaded_documents = {}
    
    # --- Load all PDF files ---
    print("\n" + "-"*80)
    print("Loading PDF Files")
    print("-"*80)
    for pdf_file in all_files['pdf']:
        print(f"\n>>> Processing: {pdf_file}")
        content = load_pdf_data(pdf_file)
        if content:
            loaded_documents[pdf_file] = content
            print(f"✓ Successfully loaded: {pdf_file}")
        else:
            print(f"✗ Failed to load: {pdf_file}")
    
    # --- Load all DOCX files ---
    print("\n" + "-"*80)
    print("Loading DOCX Files")
    print("-"*80)
    for docx_file in all_files['docx']:
        print(f"\n>>> Processing: {docx_file}")
        content = load_docx_data(docx_file)
        if content:
            loaded_documents[docx_file] = content
            print(f"✓ Successfully loaded: {docx_file}")
        else:
            print(f"✗ Failed to load: {docx_file}")
    
    # --- Load all Excel files ---
    print("\n" + "-"*80)
    print("Loading Excel Files")
    print("-"*80)
    for excel_file in all_files['excel']:
        print(f"\n>>> Processing: {excel_file}")
        content = load_excel_or_csv(excel_file)
        if content:
            loaded_documents[excel_file] = content
            print(f"✓ Successfully loaded: {excel_file}")
        else:
            print(f"✗ Failed to load: {excel_file}")
    
    # --- STEP 5: Print final summary and previews ---
    print("\n" + "="*80)
    print("STEP 3: Loading Results Summary")
    print("="*80)
    print(f"\nSuccessfully loaded: {len(loaded_documents)} out of {sum(len(v) for v in all_files.values())} files")
    
    # --- STEP 6: Show preview of each loaded document ---
    print("\n" + "="*80)
    print("STEP 4: Content Previews (First 300 characters of each file)")
    print("="*80)
    
    for filename, content in loaded_documents.items():
        print("\n" + "-"*80)
        print(f"FILE: {filename}")
        print(f"LENGTH: {len(content)} characters")
        print("-"*80)
        print(content[:300])  # First 300 characters
        print("...")
    
    # --- STEP 7: Return the loaded data ---
    # This dictionary can be used later for RAG indexing
    return loaded_documents


# ==============================================================================
# PREPROCESSING FUNCTION: Apply normalization and chunking to all documents
# ==============================================================================

def preprocess_all_documents(loaded_documents, chunk_size=800, overlap=100):
    """
    Apply preprocessing (normalize + chunk) to all loaded documents.
    
    WHY PREPROCESS?
    - Normalize: Clean text for consistent retrieval
    - Chunk: Split into pieces that fit embedding models
    
    PARAMETERS:
        loaded_documents (dict): Documents from find_and_load_all_documents()
        chunk_size (int): Target chunk size in characters (default: 800)
        overlap (int): Overlap between chunks in characters (default: 100)
        
    RETURNS:
        dict: Filename → list of chunks with metadata
    """
    
    print("\n" + "="*80)
    print("STEP 5: Preprocessing All Documents")
    print("="*80)
    print(f"\nPreprocessing settings:")
    print(f"  - Chunk size: {chunk_size} characters")
    print(f"  - Overlap: {overlap} characters")
    print(f"  - Normalization: lowercase, whitespace cleaning, unicode normalization")
    print(f"  - Sentence boundary preservation: enabled")
    
    # Dictionary to store all processed chunks
    # Key = filename, Value = list of chunk dictionaries
    all_chunks = {}
    total_chunks = 0
    
    # Process each document
    for filename, raw_text in loaded_documents.items():
        print(f"\n>>> Preprocessing: {filename}")
        print(f"    Raw text length: {len(raw_text)} characters")
        
        # Apply preprocessing pipeline
        chunks = preprocess_document(
            raw_text,
            normalize=True,
            chunk=True,
            chunk_size=chunk_size,
            overlap=overlap
        )
        
        # Store chunks with source filename
        # Add filename to each chunk for tracking
        for chunk in chunks:
            chunk['source_file'] = filename
        
        all_chunks[filename] = chunks
        total_chunks += len(chunks)
        
        print(f"    Created {len(chunks)} chunks")
        print(f"    ✓ Preprocessing complete")
    
    # Summary
    print("\n" + "="*80)
    print("PREPROCESSING SUMMARY")
    print("="*80)
    print(f"\nTotal documents processed: {len(loaded_documents)}")
    print(f"Total chunks created: {total_chunks}")
    print(f"Average chunks per document: {total_chunks / max(1, len(loaded_documents)):.1f}")
    
    return all_chunks


# ==============================================================================
# DEMO FUNCTION: Show before/after preprocessing examples
# ==============================================================================

def show_preprocessing_examples(loaded_documents, all_chunks):
    """
    Display before/after examples of preprocessing using the
    user-friendly display function from preprocess.py
    
    Shows what happens to the text during normalization and chunking.
    This helps understand the preprocessing decisions.
    """
    
    print("\n" + "="*80)
    print("STEP 6: Detailed Preprocessing Examples")
    print("="*80)
    
    # Pick 2-3 documents to show detailed examples
    # Prefer different document types for variety
    example_files = []
    
    # Try to get one of each type
    for filename in loaded_documents.keys():
        if filename.endswith('.pdf') and not any(f.endswith('.pdf') for f in example_files):
            example_files.append(filename)
        elif filename.endswith('.docx') and not any(f.endswith('.docx') for f in example_files):
            example_files.append(filename)
        elif filename.endswith('.xlsx') and not any(f.endswith('.xlsx') for f in example_files):
            example_files.append(filename)
        
        # Stop after 2 examples to avoid too much output
        if len(example_files) >= 2:
            break
    
    # If we didn't get 2, just take the first 2
    if len(example_files) < 2:
        example_files = list(loaded_documents.keys())[:2]
    
    # Show detailed preprocessing results for each example
    for filename in example_files:
        raw_text = loaded_documents[filename]
        chunks = all_chunks[filename]
        
        # Use the new user-friendly display function
        show_preprocessing_results(
            original_text=raw_text,
            processed_chunks=chunks,
            document_name=filename,
            show_original_sample=True,
            show_normalized_sample=True,
            show_chunk_samples=True,
            max_sample_length=400,
            max_chunks_to_show=2  # Show 2 chunks per document
        )


# ==============================================================================
# MAIN TEST PIPELINE
# ==============================================================================

def run_full_pipeline():
    """
    Complete test pipeline:
    1. Find all documents
    2. Load all documents
    3. Preprocess all documents (normalize + chunk)
    4. Show examples
    
    This demonstrates the complete data preparation for RAG.
    """
    
    print("="*80)
    print("FULL DATA PREPARATION PIPELINE TEST")
    print("="*80)
    print("\nThis will:")
    print("  1. Discover all PDF, DOCX, and Excel files")
    print("  2. Load raw text from each document")
    print("  3. Normalize and clean the text")
    print("  4. Chunk text into retrieval-ready pieces")
    print("  5. Show before/after examples")
    
    # STEP 1 & 2: Find and load documents
    loaded_documents = find_and_load_all_documents()
    
    # STEP 3: Preprocess all documents
    all_chunks = preprocess_all_documents(
        loaded_documents,
        chunk_size=800,   # 800 characters ≈ 150-200 tokens
        overlap=100       # 100 characters overlap
    )
    
    # STEP 4: Show preprocessing examples
    show_preprocessing_examples(loaded_documents, all_chunks)
    
    # Final summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    total_chunks = sum(len(chunks) for chunks in all_chunks.values())
    print(f"\n✓ Loaded {len(loaded_documents)} documents")
    print(f"✓ Created {total_chunks} preprocessed chunks")
    print(f"✓ All chunks are ready for:")
    print("    - Embedding (convert to vectors)")
    print("    - Indexing (store in vector database)")
    print("    - Retrieval (semantic search)")
    print("    - RAG pipeline (retrieval + generation)")
    print("\n" + "="*80)
    
    return loaded_documents, all_chunks


# ==============================================================================
# RUN THE TEST
# ==============================================================================

if __name__ == "__main__":
    # Execute the FULL pipeline (loading + preprocessing)
    loaded_documents, all_chunks = run_full_pipeline()
