import pymupdf  # PyMuPDF
import pypdf
import re
import os
from pathlib import Path
from metadata_handler import update_metadata_corrections

def extract_pdf_metadata(pdf_path):
    """Extracts metadata (title, author, date) from a PDF file."""
    title, author, date_published = "Unknown", "Unknown", "Unknown"

    try:
        with open(pdf_path, "rb") as f:
            pdf_reader = pypdf.PdfReader(f)
            metadata = pdf_reader.metadata or {}

            if metadata:
                # Print all metadata keys for debugging
                for key, value in metadata.items():
                    print(f"   pypdf* {key}: {value}")

            title = metadata.get("/Title", "Unknown")
            author = metadata.get("/Author", "Unknown")
            date_published = metadata.get("/CreationDate", "Unknown")
            keywords = metadata.get("/Keywords", "Unavailable")
            
            # format date correctly - YYYY-MM-DD format
            if date_published.startswith("D:"):
                date_published = f"{date_published[2:6]}-{date_published[6:8]}-{date_published[8:10]}"
    except Exception as e:
        print(f"Error extracting metadata with PyPDF: {e}")

    return title.strip(), author.strip(), date_published.strip(), keywords.strip()


def pymupdf_extract_pdf_metadata(pdf_path):
    """Extracts metadata (title, author, date) from a PDF file."""
    title, author, date_published = "Unknown", "Unknown", "Unknown"

    try:
        with open(pdf_path, "rb") as f:
            pdf_reader = pymupdf.open(f)
            metadata = pdf_reader.metadata or {}

            if metadata:
                # Print all metadata keys for debugging
                for key, value in metadata.items():
                    print(f"   pymupdf* {key}: {value}")

            title = metadata.get("/title", "Unknown")
            author = metadata.get("/author", "Unknown")
            date_published = metadata.get("/creationDate", "Unknown")
            keywords = metadata.get("/keywords", "Unavailable")
            
            # format date correctly - YYYY-MM-DD format
            if date_published.startswith("D:"):
                date_published = f"{date_published[2:6]}-{date_published[6:8]}-{date_published[8:10]}"
    except Exception as e:
        print(f"Error extracting metadata with PyPDF: {e}")

    return title.strip(), author.strip(), date_published.strip(), keywords.strip()


def adjust_page_range(page_range, true_page_1):
    """
    Adjusts the page range based on a manually specified 'true' page 1.
    - If true_page_1 = 15 and requested range is (1, 12), 
      actual PDF pages are (15, 26).
    """
    if not page_range or true_page_1 is None:
        return None  # No adjustments needed

    start, end = page_range

    # Compute actual PDF pages (zero-indexed for fitz)
    adjusted_start = (start - 1) + (true_page_1 - 1)
    adjusted_end = (end - 1) + (true_page_1 - 1)

    return (adjusted_start, adjusted_end)


def extract_pdf_full_text(pdf_path, page_range=None, true_page_1=None, header_footer_margin=30, debug=False):
    """
    Extracts all text from a PDF as one continuous stream.
    
    Parameters:
      pdf_path (str): Path to the PDF file.
      page_range (tuple): Optional tuple (start, end) as viewer page numbers.
      true_page_1 (int): If using page_range, the viewer's page number corresponding to the first "real" content page.
      header_footer_margin (float): Vertical margin (points) to ignore blocks near the top/bottom (headers/footers).
      debug (bool): If True, prints debug information.
    
    Returns:
      str: The full text extracted from the PDF.
    """
    doc = pymupdf.open(pdf_path)
    adjusted = adjust_page_range(page_range, true_page_1) if (page_range and true_page_1) else None
    if adjusted:
        page_nums = range(adjusted[0], adjusted[1] + 1)
    else:
        page_nums = range(len(doc))
    
    full_text = ""
    stop_processing = False
    for page_num in page_nums:
        if stop_processing:
            break
        page = doc[page_num]
        page_dict = page.get_text("dict")
        page_height = page.rect.height
        if debug:
            print(f"\nProcessing page {page_num} (height: {page_height})")
        for block in page_dict["blocks"]:
            bbox = block["bbox"]
            # Skip header/footer blocks based on vertical margins.
            if bbox[1] < header_footer_margin or bbox[3] > (page_height - header_footer_margin):
                if debug:
                    block_debug_text = ""
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                block_debug_text += text + " "
                    block_debug_text = block_debug_text.strip()
                    print(f"Skipping block with bbox {bbox}. Text: '{block_debug_text}'")
                continue
            # Combine text from all spans in the block.
            block_text = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        block_text += text + " "
            block_text = block_text.strip()
            if not block_text:
                continue
            # Stop processing if the block's text exactly equals "references".
            if block_text.lower().strip() == "references":
                if debug:
                    print(f"Encountered 'References' block: '{block_text}'. Stopping processing.")
                stop_processing = True
                break
            # Append the block text to the full text.
            full_text += block_text + " "
        full_text += "\n"  # Separate pages with a newline.
    return full_text.strip()


def scrape_pdf(pdf_path, page_range=None, true_page_1=None):
    """
    Extracts metadata and structured text from a PDF.
    - Books: Uses true page 1 and page range
    - Research Papers: Extracts full text as sections
    """
    title, author, date_published, keywords = extract_pdf_metadata(pdf_path)
    #title, author, date_published, keywords = pymupdf_extract_pdf_metadata(pdf_path)
    text = extract_pdf_full_text(pdf_path, page_range, true_page_1, header_footer_margin=33.5, debug = True)

    # title can't be unknown, to not override. fallback to file name
    if title == 'Unknown':
        title = Path(pdf_path).stem

    # add a warning if no text was scraped
    if not text.strip():
        print(f"Warning: No text extracted from {pdf_path}")
        flag = "empty_text"
    else:
        flag = ""

    return {
        "document_type": "book" if page_range else "research_paper",
        "title": title,
        "author": author,
        "source": pdf_path,
        "date_published": date_published,
        "keywords": keywords,
        "flag": flag,
        "text": text
    }

