# ==============================================================================
# FILE: load_csv.py
# PURPOSE: Load Excel (.xlsx, .xls) and CSV files into text format for RAG
# ==============================================================================

# --- Import Libraries ---
import pandas as pd  # Pandas helps us read Excel and CSV files easily
import os           # OS helps us check if files exist

# --- Main Loading Function ---
def load_excel_or_csv(file_path):
    """
    This function loads an Excel file (.xlsx or .xls) OR a CSV file.
    It reads all the data and converts it into text format.
    
    WHY? For RAG (Retrieval-Augmented Generation), we need text that an AI
    can search through. Spreadsheets have rows and columns, so we convert
    each row into a readable sentence like:
    "Column1: Value1 | Column2: Value2 | Column3: Value3"
    
    PARAMETERS:
        file_path (str): Path to the Excel or CSV file
        
    RETURNS:
        str: All the data as text, or None if there's an error
    """
    
    print(f"\n--- Loading file: {file_path} ---")
    
    # --- STEP 1: Check if the file exists ---
    if not os.path.exists(file_path):
        print(f"!!! ERROR: File not found: {file_path} !!!")
        return None
    
    try:
        # --- STEP 2: Detect file type and load accordingly ---
        
        # Get the file extension (like .xlsx, .xls, .csv)
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.csv':
            # --- Load CSV File ---
            # DECISION: We read CSV files with pandas.read_csv()
            # dtype=str means treat everything as text (no automatic number conversion)
            # This keeps data consistent and avoids precision issues
            print("Detected CSV file. Loading with pandas.read_csv()...")
            df = pd.read_csv(file_path, dtype=str)
            
        elif file_extension in ['.xlsx', '.xls']:
            # --- Load Excel File ---
            # DECISION: We use pandas.read_excel() for Excel files
            # sheet_name=0 means read the FIRST sheet only
            # If you need multiple sheets, you can change this later
            # dtype=str keeps everything as text
            print(f"Detected Excel file ({file_extension}). Loading with pandas.read_excel()...")
            
            # Choose the right engine based on file type
            if file_extension == '.xlsx':
                engine = 'openpyxl'  # Modern Excel files use openpyxl
            else:
                engine = 'xlrd'      # Older .xls files use xlrd
                
            df = pd.read_excel(file_path, sheet_name=0, dtype=str, engine=engine)
            
        else:
            print(f"!!! ERROR: Unsupported file type: {file_extension} !!!")
            return None
        
        # --- STEP 3: Convert DataFrame to Text ---
        # DECISION: We convert the spreadsheet data into readable text.
        # Each row becomes one line with format: "col1: val1 | col2: val2 | ..."
        # This makes it easy for AI to understand the data structure.
        
        print(f"Successfully loaded! Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        
        # Create an empty list to store text lines
        text_lines = []
        
        # Loop through each row in the DataFrame
        for index, row in df.iterrows():
            # Create a list to hold the parts of this row
            row_parts = []
            
            # Loop through each column in this row
            for column_name in df.columns:
                value = row[column_name]
                
                # Only add if the value is not empty/null
                # pd.notna() checks if the value is NOT missing
                if pd.notna(value) and str(value).strip() != '':
                    # Format as "ColumnName: Value"
                    row_parts.append(f"{column_name}: {str(value).strip()}")
            
            # Join all parts of this row with " | " separator
            if row_parts:  # Only add non-empty rows
                row_text = " | ".join(row_parts)
                text_lines.append(row_text)
        
        # --- STEP 4: Join all rows into one big text string ---
        # Each row is separated by a newline (\n)
        full_text = "\n".join(text_lines)
        
        print(f"Converted to text: {len(text_lines)} rows")
        print("--- Loading complete! ---")
        
        return full_text
    
    except Exception as e:
        # Catch any errors that happen during loading
        print(f"!!! ERROR: Failed to load file: {e} !!!")
        return None


# --- Self-Test Block ---
# This code only runs when you execute THIS file directly (python load_csv.py)
# It won't run when you import this file in another script
if __name__ == "__main__":
    print("="*70)
    print("TESTING load_csv.py")
    print("="*70)
    
    # Test with the Excel file in your workspace
    test_file = "Inflation Calculator.xlsx"
    
    # Try both root directory and data/tables/ folder
    if os.path.exists(test_file):
        test_path = test_file
    elif os.path.exists(os.path.join("data/tables", test_file)):
        test_path = os.path.join("data/tables", test_file)
    elif os.path.exists(os.path.join("../../data/tables", test_file)):
        test_path = os.path.join("../../data/tables", test_file)
    else:
        print(f"!!! Could not find {test_file} in data/tables/ folder !!!")
        test_path = None
    
    if test_path:
        # Load the file
        content = load_excel_or_csv(test_path)
        
        # If successful, print a preview
        if content:
            print("\n" + "="*70)
            print("PREVIEW: First 500 characters")
            print("="*70)
            print(content[:500])
            print("\n...")
            print(f"\nTotal length: {len(content)} characters")
