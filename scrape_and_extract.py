import requests
from newspaper import Article
from bs4 import BeautifulSoup
import re
import os
import fitz  # PyMuPDF
import pypdf


def get_html_soup(url):
    """Fetches HTML content and returns a BeautifulSoup object."""
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    if response.status_code == 200:
        return BeautifulSoup(response.text, "html.parser")
    else:
        print(f" Failed to fetch page: {url}")
        return None

def extract_author(article, soup):
    """Attempts to extract the author using Newspaper3k first, then BeautifulSoup."""
    author = article.authors if article.authors else []
    
    # If Newspaper3k fails, try manual extraction with BeautifulSoup
    if not author and soup:
        author_tag = soup.find("meta", attrs={"name": "author"})  # Many sites store author here
        if author_tag:
            author = [author_tag["content"].strip()]
        else:
            author_div = soup.find("div", class_=re.compile(r"author", re.IGNORECASE))
            if author_div:
                author = [author_div.text.strip()]

    return author[0] if author else "Unknown"

def extract_date(article, soup):
    """Attempts to extract the publication date using Newspaper3k first, then BeautifulSoup."""
    date_published = str(article.publish_date) if article.publish_date else "Unknown"

    if date_published == "Unknown" and soup:
        time_tag = soup.find("time")  # Many sites use <time> tag
        if time_tag:
            date_published = time_tag.text.strip()
        else:
            date_meta = soup.find("meta", attrs={"property": "article:published_time"})  # OpenGraph format
            if date_meta:
                date_published = date_meta["content"]

    return date_published if date_published != "" else "Unknown"

def scrape_article(url):
    """Scrapes an online article, extracts metadata and text, and structures it in JSON format."""

    soup = get_html_soup(url)
    article = Article(url)
    article.download()
    article.parse()

    # Extract metadata
    author = extract_author(article, soup)
    date_published = extract_date(article, soup)

    # Prepare data structure
    data = {
        "document_type": "blog_post",
        "title": article.title if article.title else "Untitled",
        "author": author,
        "source_url": url,
        "date_published": date_published,
        "sections": []
    }

    # Extract subheadings & text sections
    subheadings = soup.find_all(["h2", "h3"]) if soup else []
    content = article.text.split("\n")  
    section = {"subheading": "Introduction", "text": ""}

    for paragraph in content:
        if any(heading.text.strip() in paragraph for heading in subheadings):
            data["sections"].append(section)
            section = {"subheading": paragraph.strip(), "text": ""}
        else:
            section["text"] += paragraph.strip() + " "

    data["sections"].append(section)

    return data

# extract from pdf

def extract_pdf_pages(pdf_path, output_path, logical_page_range=None, manual_offset=None):
    """Extracts pdf pages based on a manually specified 'true' Page 1 offset if provided."""

    # Ensure output folder exists
    output_folder = os.path.dirname(output_path)
    os.makedirs(output_folder, exist_ok=True)

    doc = fitz.open(pdf_path)

    # Default to raw page numbers if no offset is specified
    page_offset = manual_offset if manual_offset is not None else 0

    # If no specific page range is given, extract all pages
    if logical_page_range is None:
        logical_page_range = range(0, len(doc))

    # Adjust logical page numbers to actual PDF raw pages
    adjusted_range = [p + page_offset for p in logical_page_range if 0 <= (p + page_offset) < len(doc)]
    
    if not adjusted_range:
        print(f" No valid pages found for {pdf_path} after adjustment.")
        return

    new_pdf = fitz.open()
    for page_num in adjusted_range:
        new_pdf.insert_pdf(doc, from_page=page_num, to_page=page_num)

    new_pdf.save(output_path)
    print(f" Extracted logical pages {list(logical_page_range)} (adjusted to {list(adjusted_range)}) saved to {output_path}")
    return



def extract_pdf_metadata(pdf_path):
    """Extracts title, author, and date from the original PDF metadata using PyPDF, with a fallback to PyMuPDF."""
    
    title, author, date_published = "Unknown", "Unknown", "Unknown"

    print(f"\n Extracting metadata from ORIGINAL PDF: {pdf_path}")

    # **🔹 Try to extract metadata using PyPDF**
    try:
        with open(pdf_path, "rb") as f:
            pdf_reader = pypdf.PdfReader(f)
            metadata = pdf_reader.metadata

            print(f" Raw Metadata from PyPDF: {metadata}")  # Debugging output

            if metadata:
                # Print all metadata keys for debugging
                for key, value in metadata.items():
                    print(f"   🔹 {key}: {value}")

                title = metadata.get("/Title", title)
                author = metadata.get("/Author", author)
                date_published = metadata.get("/CreationDate", date_published)

                print(f" Extracted from PyPDF Metadata: Title='{title}', Author='{author}', Date='{date_published}'")

                # **🔹 Format the date correctly if it exists**
                if date_published and date_published.startswith("D:"):
                    date_published = f"{date_published[2:6]}-{date_published[6:8]}-{date_published[8:10]}"  # YYYY-MM-DD format
                    print(f"Formatted Date: {date_published}")

    except Exception as e:
        print(f" Error extracting metadata from {pdf_path} with PyPDF: {e}")

    # **🔹 If metadata is still missing, fallback to PyMuPDF**
    if title == "Unknown" or author == "Unknown":
        print(" Metadata missing, searching first pages with PyMuPDF...")
        doc = fitz.open(pdf_path)
        for page in doc[:3]:  # Check first 3 pages for metadata
            text = page.get_text("text")

            # **🔹 Extract Title (assumes it's in large font)**
            title_match = re.search(r"(?<=\n)\s*([A-Z][^\n]{5,100})\s*(?=\n)", text)  
            if title_match and title == "Unknown":
                title = title_match.group(1).strip()
                print(f" Found Title in Document: {title}")

            # **🔹 Extract Author (look for "By")**
            author_match = re.search(r"(?i)by\s+([A-Z][A-Za-z\s]+)", text)
            if author_match and author == "Unknown":
                author = author_match.group(1).strip()
                print(f" Found Author in Document: {author}")

            # **🔹 Extract Date (common year format)**
            date_match = re.search(r"\b(20\d{2}|19\d{2})\b", text)  
            if date_match and date_published == "Unknown":
                date_published = date_match.group(1)
                print(f" Found Date in Document: {date_published}")

    return title.strip(), author.strip(), date_published.strip()

def scrape_pdf(pdf_path, page_range=None, course_name="Unknown Course"):
    """Extracts text and metadata from a PDF, assigns document type, and removes references for research papers."""
    
    print(f"\n Processing PDF: {pdf_path}")

    # **🔹 Extract metadata from the original (full) PDF, not the clipped version**
    title, author, date_published = extract_pdf_metadata(pdf_path)

    doc = fitz.open(pdf_path)
    document_type = "book" if page_range else "research_paper"
    print(f" Document Type: {document_type}")

    extracted_text = []
    references_found = False

    for page in doc:
        text = page.get_text("text")

        if document_type == "research_paper" and re.search(r"\bReferences\b", text, re.IGNORECASE):
            print(f" Stopped extraction at 'References' section in {pdf_path}")
            references_found = True
            break  # Stop extracting further pages

        extracted_text.append(text.strip())

    if document_type == "book":
        return {
            "document_type": document_type,
            "title": title,
            "author": author,
            "source_url": pdf_path,
            "date_published": date_published,
            "full_text": "\n".join(extracted_text),
            "course": course_name
        }

    sections = []
    for i, text in enumerate(extracted_text):
        sections.append({
            "subheading": f"Section {i+1}",
            "text": text
        })

    print(f" Extracted {len(sections)} sections from research paper.")

    return {
        "document_type": document_type,
        "title": title,
        "author": author,
        "source_url": pdf_path,
        "date_published": date_published,
        "sections": sections,
        "course": course_name
    }
