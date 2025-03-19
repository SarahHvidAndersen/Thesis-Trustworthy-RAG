import re

def clean_filename(title):
    """Removes invalid filename characters and replaces spaces with underscores."""
    return re.sub(r'[<>:"/\\|?*]', '', title).replace(' ', '_')