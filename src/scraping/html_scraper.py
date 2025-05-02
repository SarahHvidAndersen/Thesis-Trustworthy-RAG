import requests
from newspaper import Article
from bs4 import BeautifulSoup
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from logging.logging_config import logger

def get_html_soup(url):
    """Fetches HTML content from a URL and returns a BeautifulSoup object for parsing."""
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code == 200:
            return BeautifulSoup(response.text, "html.parser")
        else:
            logger.warning(f"Error {response.status_code}: Failed to fetch {url}")
            return None
    except requests.RequestException as e:
        logger.warning(f"Request failed: {e}")
        return None

def extract_author(article, soup):
    """Extracts author from an article using Newspaper3k and BeautifulSoup as fallback."""
    if article.authors:
        return article.authors[0]  

    if soup:
        author_meta = soup.find("meta", attrs={"name": "author"})
        if author_meta:
            return author_meta["content"].strip()
        author_div = soup.find("div", class_=re.compile(r"author", re.IGNORECASE))
        if author_div:
            return author_div.text.strip()

    return "Unknown"

def extract_date(article, soup):
    """Extracts publication date using Newspaper3k and fallback parsing from HTML."""
    if article.publish_date:
        return str(article.publish_date)  

    if soup:
        time_tag = soup.find("time")
        if time_tag:
            return time_tag.text.strip()

        date_meta = soup.find("meta", attrs={"property": "article:published_time"})
        if date_meta:
            return date_meta["content"]

    return "Unknown"

def extract_html_text(soup, debug=False):
    """
    Extracts the full text from the HTML soup by joining paragraphs,
    but skips any paragraph that contains (as a whole word) one of the ignore keywords.
    """
    paragraphs = []
    # Define a set of keywords that, if present as a whole word in a paragraph,cause that paragraph to be skipped.
    ignore_keywords = {"cookie", "cookies", "subscribe", "references", "privacy", "footer", "sign in", "sign up"}
    
    # Compile a regex that matches any ignore keyword as a whole word.
    pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, ignore_keywords)) + r')\b', re.IGNORECASE)
    
    for p in soup.find_all("p"):
        text = p.get_text(strip=True)
        if text:
            # If any whole word in the paragraph exactly matches an ignore keyword, skip it.
            if pattern.search(text):
                if debug:
                    logger.debug(f"Skipping paragraph due to keyword match: '{text}'")
                continue
            paragraphs.append(text)
    
    return " ".join(paragraphs)

def scrape_au_course(url):
    """
    Uses Selenium to render the JavaScript on an AU course page,
    then extracts the course description text.
    """
    # Set up headless Chrome
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--log-level=3")  # Only log errors
    
    # chromedriver should load automatically, otherwise make sure you have the correct one in your PATH.
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    
    # Wait for content to load
    time.sleep(5)
    
    html = driver.page_source
    driver.quit()
    
    soup = BeautifulSoup(html, "html.parser")
    
    # Find the container with the course description
    container = soup.find("main")
    if container:
        logger.info("Found course description container.")
        return container.get_text(separator="\n", strip=True)
    else:
        logger.debug("Course description container not found. Returning full page text.")
        return soup.get_text(separator="\n", strip=True)

def scrape_html_standard(url):
    """
    Extracts metadata and content from an online article or blog post.
    """
    
    soup = get_html_soup(url)
    article = Article(url)
    
    try:
        article.download()
        article.parse()
    except Exception as e:
        logger.warning(f"Error parsing article: {e}")
        return None
    
    text = extract_html_text(soup, debug = True)

    if not text.strip():
        logger.warning(f"Warning: No text extracted from {url}")
        flag = "empty_text"
    else:
        flag = ""

    return {
        "document_type": "online_article",
        "title": article.title or url.split("/")[-1].replace("-", " ").title(),  # Guess title from URL
        "author": extract_author(article, soup),
        "source": url,
        "date_published": extract_date(article, soup),
        "flag": flag,
        "text": text,
    }


def scrape_html(url):
    """
    General HTML scraper that chooses the appropriate extraction method based on the URL.
    For AU course pages (identified by "kursuskatalog.au.dk" in the URL), it uses a Selenium-based dynamic scraper.
    Otherwise, it uses the standard scraper.
    """
    if "kursuskatalog.au.dk" in url:
        text = scrape_au_course(url)

        if not text.strip():
            logger.warning(f"Warning: No text extracted from {url}")
            flag = "empty_text"
        else:
            flag = ""

        logger.info("Detected AU course page; using dynamic scraper")

        data = {
            "document_type": "AU_course_page",
            "title" : url.split("/")[-1].replace("-", " ").title(),  # Guess title from URL
            "author": "Aarhus University",
            "source": url,
            "flag": flag,
            "text": text
        }

        return data
    
    else:
        soup = get_html_soup(url)
        article = Article(url)
        
        try:
            article.download()
            article.parse()
        except Exception as e:
            logger.warning(f"Error parsing article: {e}")
            return None
        
        text = extract_html_text(soup, debug = True)

        if not text.strip():
            logger.warning(f"Warning: No text extracted from {url}")
            flag = "empty_text"
        else:
            flag = ""

        data = {
            "document_type": "online_article",
            "title": article.title or url.split("/")[-1].replace("-", " ").title(),  # Guess title from URL
            "author": extract_author(article, soup),
            "source": url,
            "date_published": extract_date(article, soup),
            "flag": flag,
            "text": text
        }
        return data


