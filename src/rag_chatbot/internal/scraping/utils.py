import re
import os
import hashlib
import json
from logging_scripts.logging_config import logger

def clean_filename(title):
    """Removes invalid filename characters and replaces spaces with underscores."""
    return re.sub(r'[<>:"/\\|?*]', '', title).replace(' ', '_')


def get_stable_filename(directory, title, source, extension=".json", page_range=None):
    """
    Returns a stable filename for the scraped data.
    
    For online articles (when page_range is None):
      - It creates a candidate filename from the clean title.
      - If that file already exists, it reads its content and checks the stored "source".
         * If the "source" matches, it returns that filename (so that it is updated).
         * If it does not match, it appends a short MD5 hash of the source.
    
    For PDFs (when page_range is provided):
      - It appends the page range to the title and returns that filename.
    """
    # Start with a cleaned title.
    base_filename = clean_filename(title).lower()
    base_filename = base_filename.replace(" ", "_").lower()
    
    if page_range:
        # For PDFs, append the page range
        page_range = str(page_range).replace(',', '-')
        page_range_str = f"_page-{page_range.translate(str.maketrans('', '', "() "))}" 
        base_filename += page_range_str
        filepath = os.path.join(directory, f"{base_filename}{extension}")
        return filepath
    else:
        # For online articles, create a candidate filename.
        candidate = os.path.join(directory, f"{base_filename}{extension}")
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # If the source matches, use the candidate.
                if data.get("source") == source:
                    logger.debug(f"source from loaded file: {data.get("source")}")
                    logger.debug(f"source from currently processing file: {source}")
                    logger.debug(f"and candidate is: {candidate}")
                    return candidate
                else:
                    # Otherwise, append a hash to differentiate.
                    url_hash = hashlib.md5(source.encode("utf-8")).hexdigest()[:8]
                    candidate = os.path.join(directory, f"{base_filename}_{url_hash}{extension}")
                    return candidate
            except Exception:
                # In case of error reading the file, fallback to using the hash.
                url_hash = hashlib.md5(source.encode("utf-8")).hexdigest()[:8]
                candidate = os.path.join(directory, f"{base_filename}_{url_hash}{extension}")
                return candidate
        else:
            return candidate