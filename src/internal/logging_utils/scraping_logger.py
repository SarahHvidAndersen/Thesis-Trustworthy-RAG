import logging
from logging import StreamHandler
import sys

# Create a root logger and set its level to DEBUG to capture all messages.
logger = logging.getLogger('scraping_courses_logger')
logger.setLevel(logging.DEBUG)


# Prevent multiple handlers if the logger is configured multiple times
if not logger.handlers:
    # Create a file handler to log to a file
    #file_handler = logging.FileHandler("processed_syllabi/scraping_courses_logger.txt", mode='w', encoding='utf-8')
    file_handler = logging.FileHandler('data/processed_syllabi/scraping_courses_log_file.log')
    file_handler.setLevel(logging.DEBUG)

    # Define log format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Create a stream handler that logs only INFO and above (so debug messages aren't printed to terminal)
    stream_handler = StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(stream_formatter)

    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    # Add both handlers to the root logger.
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)




