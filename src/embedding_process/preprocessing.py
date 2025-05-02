import re
import unicodedata
import json

def clean_text(text):
    """
    Cleans the input text by:
      - Normalizing Unicode using NFKC.
      - Removing control characters (except newline and tab).
      - Removing unwanted symbols (e.g., ∗, ‡, §, ¶, ‖, etc.).
      - Removing or collapsing sequences of dots.
      - Collapsing multiple whitespace characters.
    
    Adjust the regexes below to either collapse or remove unwanted patterns.
    """
    # Normalize Unicode (NFKC to standardize characters)
    text = unicodedata.normalize('NFKC', text)
    
    # Remove control characters (except newline \n and tab \t)
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    
    # Remove carriage returns to standardize newlines.
    text = text.replace('\r', '')
    
    # Remove unwanted symbols: list them inside square brackets.
    unwanted_symbols = r'[‡§¶‖]'
    text = re.sub(unwanted_symbols, '', text)

    # words were often split like this: param- eter
    text = re.sub(r'[a-zA-Z]- ', '', text)
    
    # Remove standalone sequences of 3 or more dots (with optional spaces between).
    text = re.sub(r'(?:\.\s*){3,}', '', text)
    
    # remove DOI patterns? maybe, doesn't work too well, should be more greedy in that case
    #text = re.sub(r'doi:\s*\S+', '', text, flags=re.IGNORECASE)

    #remove spaces that occur before or after removed dots (if any)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

if __name__ == "__main__":
    # Test the cleaning function on a sample file.

    sample_file = r"fprocessed_syllabi\Adv_cognitive_modelling\scraped_data\bayesian_workflow.json"

    try:
        with open(sample_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Get the raw text
        raw_text = data.get("text", "")

    except FileNotFoundError:
        print(f"File '{sample_file}' not found. Please create one for testing.")
        raw_text = """
         At trial  inserting 0- ToM’s learning rule ( Eq 3 ) into. we also rec- ommend looking at the correlation between the recovered parameters themselves. If the simulation parameters are uncorrelated with one another, correlation between the recovered parameters is an indication that the parameters in the model are trading off against one another ( Daw, 2011 ). Such trade-offs can sometimes be avoided by reparameterizing the model (e.g. Otto et al., 2013 ) or redesigning the experiment. Sometimes, however, such trade-offs are unavoidable",
        """
    
    cleaned = clean_text(raw_text)
    print("Cleaned Text:")
    print(cleaned)