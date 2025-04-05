import re
import json
import fitz  # pymupdf

def modify_toc_with_ranges(input_file = './doc_data/toc.json', output_file = './doc_data/toc.json'):
    with open(input_file, 'r', encoding='utf-8') as f:
        toc = json.load(f)
    
    for i in range(len(toc) - 1):
        toc[i]["page_range"] = [toc[i]["page"], toc[i + 1]["page"] - 1]
    
    toc[-1]["page_range"] = [toc[-1]["page"], toc[-1]["page"] + 1]  # Assume last section spans 2 pages
    
    for entry in toc:
        entry.pop("page", None)  # Remove single-page references
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(toc, f, indent=4, ensure_ascii=False)
    
    print(f"Modified TOC saved to {output_file}")

def extract_toc(pdf_path, output_json = './doc_data/toc.json'):
    """Extracts the Table of Contents (ToC) from a PDF and saves it as a JSON file."""
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    
    # Convert ToC into a structured format
    structured_toc = [
        {"level": entry[0], "title": entry[1], "page": entry[2]}
        for entry in toc
    ]

    # Save to JSON file
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(structured_toc, f, indent=4, ensure_ascii=False)
    
    print(f"ToC extracted and saved to {output_json}")

def parse_toc(json_file = './doc_data/toc.json'):
    """Reads the ToC JSON file and reconstructs it."""
    with open(json_file, "r", encoding="utf-8") as f:
        toc_data = json.load(f)
    
    return toc_data
    
def clean_text(text: str) -> str:
    """
    Clean extracted text to remove unwanted characters and normalize spacing.
    """
    text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with space
    text = re.sub(r'(?<=\w)-\s+', '', text)  # Remove hyphenated line breaks
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    return text

def extract_section_metadata(text: str) -> str:
    """
    Extract potential section titles based on common handbook formatting (e.g., numbered sections).
    """
    match = re.search(r'(\b\d+\.\d+\b.*)', text)  # Example: "1.1 Code of Conduct"
    return match.group(1) if match else "Unknown Section"   
