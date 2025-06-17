import os
import json
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json

def extract_pdf_content(pdf_path, output_dir):
    """
    Extract tables and text from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str): Directory to save the output files
    
    Returns:
        tuple: (tables_html, text_content) where tables_html is a list of HTML tables
               and text_content is a list of text elements
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract elements from PDF using hi_res strategy and YOLOX model for better table detection
    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        infer_table_structure=True,
        model_name="yolox"
    )
    
    # Save raw elements to JSON for reference
    json_path = os.path.join(output_dir, "raw_elements.json")
    elements_to_json(elements, filename=json_path)
    
    # Separate tables and text
    tables_html = []
    text_content = []
    
    # Process the elements
    for element in elements:
        if hasattr(element, 'metadata') and element.metadata.get('text_as_html'):
            # This is a table
            tables_html.append({
                'table_html': element.metadata['text_as_html'],
                'page_number': element.metadata.get('page_number')
            })
        else:
            # This is text content
            text_content.append({
                'text': str(element),
                'category': element.category if hasattr(element, 'category') else 'unknown',
                'page_number': element.metadata.get('page_number') if hasattr(element, 'metadata') else None
            })
    
    # Save tables to a separate file
    tables_path = os.path.join(output_dir, "tables.json")
    with open(tables_path, 'w', encoding='utf-8') as f:
        json.dump(tables_html, f, indent=2, ensure_ascii=False)
    
    # Save text content to a separate file
    text_path = os.path.join(output_dir, "text_content.json")
    with open(text_path, 'w', encoding='utf-8') as f:
        json.dump(text_content, f, indent=2, ensure_ascii=False)
    
    return tables_html, text_content

def main():
    # Example usage
    pdf_path = "C:/Users/rahul/OneDrive/Desktop/Moco/US Foods Invoices.pdf"  # Replace with your PDF path
    output_dir = "C:/Users/rahul/OneDrive/Desktop/Moco/result"         # Replace with desired output directory
    
    try:
        tables, text = extract_pdf_content(pdf_path, output_dir)
        
        print(f"Extraction complete!")
        print(f"Found {len(tables)} tables")
        print(f"Found {len(text)} text elements")
        print(f"Results saved in: {output_dir}")
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")

if __name__ == "__main__":
    main()