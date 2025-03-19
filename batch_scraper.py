import json
import requests
from newspaper import Article
from bs4 import BeautifulSoup
import re
import os

from scrape_and_extract import *
from scrape_and_extract import extract_pdf_pages, scrape_pdf, scrape_article


CORRECTIONS_FILE = "processed_syllabi/metadata_corrections.json"


def sanitize_filename(title):
    """Removes invalid filename characters and replaces spaces with underscores."""
    return re.sub(r'[<>:"/\\|?*]', '', title).replace(' ', '_')

def load_metadata_corrections():
    """Loads manually corrected metadata from a JSON file or creates an empty one if missing."""
    if not os.path.exists(CORRECTIONS_FILE):
        with open(CORRECTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=4, ensure_ascii=False)  # Initialize empty JSON
    with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_metadata_corrections(corrections):
    """Saves full metadata for error spotting and manual corrections."""
    with open(CORRECTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(corrections, f, indent=4, ensure_ascii=False)

def save_article_json(article_data, output_folder, metadata_corrections, course_name, chapter_title=""):
    """Saves article data as JSON and ensures books are stored only once."""
    
    title = sanitize_filename(article_data.get("title", "Untitled"))
    chapter_title_clean = sanitize_filename(chapter_title)
    filename = f"{output_folder}/{title}_{chapter_title_clean}.json"

    article_data["course"] = course_name

    # **🔹 Store Books Only Once in `metadata_corrections.json`**
    file_key = f"{course_name}/{title}"
    if article_data["document_type"] == "book" and file_key in metadata_corrections:
        print(f"🔹 Book '{title}' already logged, skipping duplicate metadata entry.")
    else:
        metadata_corrections[file_key] = {
            "course": course_name,
            "title": article_data.get("title", "Untitled"),
            "author": article_data.get("author", "Unknown"),
            "date_published": article_data.get("date_published", "Unknown")
        }
        save_metadata_corrections(metadata_corrections)
        print(f" save article json, Stored metadata for x")

    # **🔹 Save Extracted Data as JSON**
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(article_data, f, indent=4, ensure_ascii=False)

    print(f"- Successfully processed: {filename}")



def process_pdf_text_extraction(pdf_path, page_range, manual_offset, clipped_folder, output_folder, metadata_corrections, course_name):
    """Handles the text extraction from clipped PDFs and attaches correct metadata."""

    print(f"\n Extracting Text from PDF Pages: {pdf_path}")

    # **🔹 Only extract chapter title if page_range is specified**
    if page_range:
        chapter_title = extract_chapter_title(pdf_path, page_range, manual_offset)
        chapter_title_clean = sanitize_filename(chapter_title) if chapter_title else f"chapter_{min(page_range)}"
    else:
        chapter_title_clean = "full_document"  # Default for full PDFs

    extracted_pdf_name = os.path.basename(pdf_path).replace(".pdf", f"_{chapter_title_clean}_extracted.pdf")
    output_pdf = os.path.join(clipped_folder, extracted_pdf_name)

    print(f"   ➡ Extracting pages to: {output_pdf}")

    # **🔹 Extract pages if specified, otherwise process full document**
    extract_pdf_pages(pdf_path, output_pdf, page_range, manual_offset)

    # **🔹 Extract text from the clipped PDF**
    article_data = scrape_pdf(output_pdf, page_range, course_name)

    # **🔹 Attach metadata from metadata_corrections.json**
    file_key = f"{course_name}/{os.path.basename(pdf_path)}"
    if file_key in metadata_corrections:
        article_data.update(metadata_corrections[file_key])

    # **🔹 Save JSON using the same chapter-based naming convention**
    save_article_json(article_data, output_folder, metadata_corrections, course_name, chapter_title_clean)


def extract_chapter_title(pdf_path, page_range, manual_offset):
    """Extracts the most prominent heading from the first page of a specified page range."""
    
    doc = fitz.open(pdf_path)
    first_page_index = min(page_range) + (manual_offset or 0) - 1  # Adjust page offset
    
    if first_page_index >= len(doc):
        print(f" Page {first_page_index + 1} out of range for {pdf_path}.")
        return None

    page = doc[first_page_index]
    text_instances = []

    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                text_instances.append((span["size"], span["text"].strip()))

    if not text_instances:
        return None

    # **🔹 Sort text by font size (largest first) and return first non-empty line**
    text_instances.sort(reverse=True, key=lambda x: x[0])
    
    for _, text in text_instances:
        if len(text) > 5:  # Avoid short words
            print(f" Identified Chapter Title: {text}")
            return text

    return None


def process_course_syllabi(course_name):
    """Reads URLs & PDFs from materials_paths.txt, extracts metadata first, then extracts text."""
    
    raw_path = f"raw_syllabi/master_courses/{course_name}/materials_paths_test.txt"
    output_folder = f"processed_syllabi/{course_name}/"
    clipped_folder = f"{output_folder}/clipped_pdfs/"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(clipped_folder):
        os.makedirs(clipped_folder)

    # Load existing metadata corrections
    metadata_corrections = load_metadata_corrections()

    with open(raw_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            article_data = None
            page_range = None
            manual_offset = None

            # **🔹 Process Online Articles**
            if line.startswith("http"):  
                print(f"🔗 Scraping article: {line}")
                article_data = scrape_article(line)
                save_article_json(article_data, output_folder, metadata_corrections, course_name)
                continue  # Skip to next line

            # **🔹 Process PDFs (Metadata Extraction)**
            match = re.match(r"(.+?\.pdf)(?:\s+pages=(\d+-\d+))?(?:\s+offset=(\d+))?", line)
            if match:
                pdf_path = match.group(1).strip()
                if match.group(2):
                    page_range = range(*map(int, match.group(2).split("-")))
                if match.group(3):
                    manual_offset = int(match.group(3))

                print(f" Processing PDF Metadata: {pdf_path}")

                # **🔹 Extract metadata from the original PDF (before clipping)**
                title, author, date_published = extract_pdf_metadata(pdf_path)

                # **🔹 Ensure metadata is stored once**
                file_key = f"{course_name}/{os.path.basename(pdf_path)}"
                if file_key not in metadata_corrections:
                    metadata_corrections[file_key] = {
                        "course": course_name,
                        "title": title,
                        "author": author,
                        "date_published": date_published
                    }
                    save_metadata_corrections(metadata_corrections)
                    print(f" process - function, Stored metadata for {title}")

                # **🔹 Extract Pages (Text Extraction Phase)**
                process_pdf_text_extraction(pdf_path, page_range, manual_offset, clipped_folder, output_folder, metadata_corrections, course_name)


if __name__ == "__main__":
    process_course_syllabi("Human_computer_interaction")



