from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time

def scrape_au_course_dynamic(url):
    """
    Uses Selenium to render the JavaScript on an AU course page,
    then extracts the course description text.
    """
    # Set up headless Chrome
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    
    # Make sure you have the appropriate chromedriver in your PATH.
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    
    # Wait for content to load. Adjust time if necessary.
    time.sleep(5)
    
    html = driver.page_source
    driver.quit()
    
    soup = BeautifulSoup(html, "html.parser")
    
    # Try to find the container with the course description.
    # Inspect the page to adjust this selector if needed.
    container = soup.find("main")
    if container:
        print("Found course description container.")
        return container.get_text(separator="\n", strip=True)
    else:
        print("Course description container not found. Returning full page text.")
        return soup.get_text(separator="\n", strip=True)

if __name__ == "__main__":
    url = "https://kursuskatalog.au.dk/en/course/114898/Human-Computer-Interaction"
    course_text = scrape_au_course_dynamic(url)
    print("\nExtracted Course Text:")
    print(course_text)
