import json
import os
import logging
from logging_config import logger  # Import the shared logger

CORRECTIONS_FILE = r"processed_syllabi\metadata_corrections.json"

def load_metadata_corrections():
    """Loads manually corrected metadata from a JSON file or creates an empty one if missing."""
    if not os.path.exists(CORRECTIONS_FILE):
        with open(CORRECTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=4, ensure_ascii=False)  # Initialize empty JSON
    with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_metadata_corrections(metadata):
    """Saves updated metadata corrections."""
    with open(CORRECTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    return

def update_metadata_corrections(document_data):
    """
    Check if document metadata exists in corrections file.
    - If missing, add it.
    - If already exists, keep the existing manual corrections.
    """
    metadata_corrections = load_metadata_corrections()
    doc_source = document_data["source"]

    # Add document if it does not exist
    if doc_source not in metadata_corrections:
        metadata_corrections[doc_source] = {
            "title": document_data["title"],
            "author": document_data.get("author", "Unknown"),
            "date_published": document_data.get("date_published", "Unknown"),
            "document_type": document_data.get("document_type", "Unknown"),
            "source": document_data.get("source", "Unknown"),
            "flag": document_data.get("flag", "Unknown")
        }
        logger.info(f"New metadata added for: {doc_source}")

    save_metadata_corrections(metadata_corrections)

    return metadata_corrections  # Return updated metadata for logging
