import pandas as pd
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def sanitize_sheet_name(name: str) -> str:
    """
    Sanitize sheet name to comply with Excel restrictions.
    
    Args:
        name (str): Original sheet name
        
    Returns:
        str: Sanitized sheet name
    """
    # Remove invalid characters
    invalid_chars = [':', '/', '\\', '?', '*', '[', ']']
    for char in invalid_chars:
        name = name.replace(char, '_')
    
    # Truncate to 31 characters (Excel limit)
    return name[:31]

def combine_excel_sheets(file_paths: list, output_path: str) -> bool:
    """
    Combine multiple Excel files into a single workbook with error handling.
    
    Args:
        file_paths (list): List of paths to Excel files
        output_path (str): Path where the combined Excel file will be saved
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Convert output path to Path object and create directory if needed
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if output file already exists
        if output_path.exists():
            logging.warning(f"Output file {output_path} already exists. Will be overwritten.")
        
        # Create a Pandas Excel writer using XlsxWriter as the engine
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            processed_files = 0
            
            for file_path in file_paths:
                try:
                    file_path = Path(file_path)
                    
                    # Verify file exists
                    if not file_path.exists():
                        logging.error(f"File not found: {file_path}")
                        continue
                    
                    # Extract and sanitize sheet name
                    sheet_name = sanitize_sheet_name(file_path.stem)
                    
                    # Read the file into a DataFrame
                    logging.info(f"Processing file: {file_path}")
                    df = pd.read_excel(file_path)
                    
                    # Write the DataFrame to the Excel file
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    processed_files += 1
                    logging.info(f"Successfully added sheet: {sheet_name}")
                    
                except Exception as e:
                    logging.error(f"Error processing {file_path}: {str(e)}")
                    continue
            
            logging.info(f"Successfully processed {processed_files} out of {len(file_paths)} files")
            return processed_files > 0

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        return False

if __name__ == "__main__":
    # List of file paths to combine
    file_paths = [
        r"C:\Users\rahul\Downloads\Specialty_Latte_Cost_Analysis.xlsx",
        r"C:\Users\rahul\Downloads\Latte_Cost_Analysis_Corrected.xlsx",
        r"C:\Users\rahul\Downloads\Page3_Latte_Cost_Analysis.xlsx",
        r"C:\Users\rahul\Downloads\Page4_Drinks_Cost_Analysis.xlsx",
        r"C:\Users\rahul\Downloads\Page5_Drinks_Cost_Analysis.xlsx",
        r"C:\Users\rahul\Downloads\Page6_Drinks_Cost_Analysis.xlsx",
        r"C:\Users\rahul\Downloads\Page7_Drinks_Cost_Analysis.xlsx",
        r"C:\Users\rahul\Downloads\Page8_Drinks_Cost_Analysis.xlsx",
        r"C:\Users\rahul\Downloads\Page9_Drinks_Cost_Analysis.xlsx",
        r"C:\Users\rahul\Downloads\Page10_Bulk_Recipes_Cost_Analysis.xlsx"
    ]

    # Output file path
    output_path = r"C:\Users\rahul\Downloads\Combined_Drinks_Cost_Analysis.xlsx"

    # Combine the sheets
    success = combine_excel_sheets(file_paths, output_path)
    
    if success:
        logging.info(f"Successfully created combined Excel file at: {output_path}")
    else:
        logging.error("Failed to create combined Excel file")