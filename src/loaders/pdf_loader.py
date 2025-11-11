# ==============================================================================
# FILE: load_pdf.py
# PURPOSE: Load PDF files and extract all text content for RAG
# ==============================================================================

# --- Import Libraries ---
from PyPDF2 import PdfReader  # PyPDF2 helps us read PDF files
import os                      # OS helps us check if files exist

# --- Main Loading Function ---
def load_pdf_data(file_path):
    """
    This function loads text from a PDF file.
    
    WHY? PDF files can contain research papers, documents, reports, etc.
    We need to extract ALL the text so an AI can search through it.
    PDFs are made of "pages", so we read each page one by one and combine them.
    
    PREPROCESSING DECISIONS:
    1. We read page by page because PDFs are structured that way
    2. We join pages with double newlines (\n\n) to keep page boundaries clear
    3. We clean up extra whitespace that PDFs sometimes have
    
    PARAMETERS:
        file_path (str): Path to the PDF file
        
    RETURNS:
        str: All text from the PDF, or None if there's an error
    """
    
    print(f"\n--- Loading PDF file: {file_path} ---")
    
    # --- STEP 1: Check if the file exists ---
    if not os.path.exists(file_path):
        print(f"!!! ERROR: File not found: {file_path} !!!")
        return None
    
    try:
        # --- STEP 2: Open and read the PDF ---
        # DECISION: Use PyPDF2's PdfReader to open the PDF file
        # This is a reliable library for extracting text from PDFs
        print("Opening PDF with PyPDF2.PdfReader()...")
        
        reader = PdfReader(file_path)
        
        # Get the total number of pages in the PDF
        total_pages = len(reader.pages)
        print(f"PDF has {total_pages} pages")
        
        # --- STEP 3: Extract text from each page ---
        # DECISION: We loop through each page and extract its text
        # Some pages might be empty or have only images - we handle that
        
        # Create an empty list to store text from each page
        page_texts = []
        
        # Loop through each page (page numbers start at 0)
        for page_num in range(total_pages):
            # Get the page object
            page = reader.pages[page_num]
            
            # Extract text from this page
            # extract_text() is a PyPDF2 function that gets all readable text
            page_text = page.extract_text()
            
            # --- STEP 4: Clean the text ---
            # DECISION: PDFs sometimes have extra spaces, weird line breaks, etc.
            # We clean this up to make it more readable for the AI
            
            if page_text:  # Only process if there's text on this page
                # Remove extra whitespace and normalize spaces
                # strip() removes leading/trailing whitespace
                # We also replace multiple spaces with single spaces
                cleaned_text = " ".join(page_text.split())
                
                # Only add non-empty pages
                if cleaned_text:
                    page_texts.append(cleaned_text)
                    print(f"  Page {page_num + 1}: Extracted {len(cleaned_text)} characters")
            else:
                print(f"  Page {page_num + 1}: No text found (might be an image-only page)")
        
        # --- STEP 5: Combine all pages ---
        # DECISION: Join all page texts with double newlines (\n\n)
        # This keeps page boundaries somewhat visible and helps the AI
        # understand document structure
        
        if not page_texts:
            print("!!! WARNING: No text could be extracted from this PDF !!!")
            return None
        
        full_text = "\n\n".join(page_texts)
        
        print(f"Successfully extracted text from {len(page_texts)} pages")
        print(f"Total text length: {len(full_text)} characters")
        print("--- PDF loading complete! ---")
        
        return full_text
    
    except Exception as e:
        # Catch any errors that happen during PDF reading
        print(f"!!! ERROR: Failed to load PDF: {e} !!!")
        return None


# --- Self-Test Block ---
# This code only runs when you execute THIS file directly (python load_pdf.py)
if __name__ == "__main__":
    print("="*70)
    print("TESTING load_pdf.py")
    print("="*70)
    
    # Test with one of the PDF files in your workspace
    test_file = "Attention_is_all_you_need (1) (3).pdf"
    
    # Try both root directory and data/pdfs/ folder
    if os.path.exists(test_file):
        test_path = test_file
    elif os.path.exists(os.path.join("data/pdfs", test_file)):
        test_path = os.path.join("data/pdfs", test_file)
    elif os.path.exists(os.path.join("../../data/pdfs", test_file)):
        test_path = os.path.join("../../data/pdfs", test_file)
    else:
        print(f"!!! Could not find {test_file} in data/pdfs/ folder !!!")
        test_path = None
    
    if test_path:
        # Load the PDF
        content = load_pdf_data(test_path)
        
        # If successful, print a preview
        if content:
            print("\n" + "="*70)
            print("PREVIEW: First 500 characters")
            print("="*70)
            print(content[:500])
            print("\n...")
            print(f"\nTotal length: {len(content)} characters")
