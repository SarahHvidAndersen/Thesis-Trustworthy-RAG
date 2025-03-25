import os
import logging
from html_scraper import scrape_html
from pdf_processor import scrape_pdf
from metadata_handler import update_metadata_corrections
from utils import get_stable_filename
import json
import sys
from logging_config import logger

def parse_materials_paths(file_path):
    """
    Parses the materials_paths.txt file, identifying books with page ranges and true page 1 values.
    Returns a list of dictionaries containing paths and relevant metadata.
    """
    materials = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Detect book format: "path/to/book.pdf | true_page_1=5 | pages=1-50"
            parts = line.split("|")
            pdf_path = parts[0].strip()
            true_page_1, page_range = None, None

            for part in parts[1:]:
                if "true_page_1=" in part:
                    true_page_1 = int(part.split("=")[1].strip())
                elif "pages=" in part:
                    page_range = tuple(map(int, part.split("=")[1].strip().split("-")))

            materials.append({
                "path": pdf_path,
                "true_page_1": true_page_1,
                "page_range": page_range,
            })

    return materials

def save_scraped_data(course_name, document_data, page_range=None):
    """Saves extracted data into a JSON file with correct metadata and page range tracking."""

    os.makedirs(f"processed_syllabi\\{course_name}\\scraped_data", exist_ok=True)

    output_dir = f"processed_syllabi\\{course_name}\\scraped_data"

    # create clean and stable filename, that prevents articles with the same title from overwriting
    save_path = get_stable_filename(output_dir, document_data["title"], document_data["source"], ".json", page_range)

    # Ensure metadata corrections are applied & new entries are added
    metadata_corrections = update_metadata_corrections(document_data)
    if document_data["source"] in metadata_corrections:
        document_data.update(metadata_corrections[document_data["source"]])

    # Save data
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(document_data, f, indent=4, ensure_ascii=False)

    logger.info(f"Saved: {save_path}")


def process_course_syllabi(course_name):
    """
    Processes course materials, distinguishing between books, research papers, and articles.
    Saves extracted data in a structured format.
    """
    materials_file = f"raw_syllabi/master_courses/{course_name}/materials_paths.txt"
    materials = parse_materials_paths(materials_file)

    for material in materials:
        logger.info(f"Processing: {material['path']}")

        if material["path"].endswith(".pdf"):
            pdf_data = scrape_pdf(
                material["path"], 
                page_range=material["page_range"], 
                true_page_1=material["true_page_1"]
            )

            save_scraped_data(course_name, pdf_data, page_range=material["page_range"])

        elif material["path"].startswith("http"):
            html_data = scrape_html(material["path"])
            save_scraped_data(course_name, html_data)
        logger.info('')
    return


if __name__ == "__main__":

    courses = ['Human_computer_interaction', 'Natural_language_processing', 'Adv_cog_neuroscience', 'Adv_cognitive_modelling', 'Data_science', 'Decision_making']
    #courses = ['Data_science']

    for course in courses:
        os.makedirs(f"processed_syllabi/{course}", exist_ok=True)

        logger.info(f'Beginning processing of: {course}')
        
        process_course_syllabi(f"{course}")

        logger.info(f'Finished processing of: {course}')
