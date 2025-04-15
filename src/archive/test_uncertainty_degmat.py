import logging
import numpy as np
import sys
#import lm_polygraph
#from lm_polygraph.utils.deberta import Deberta
#from lm_polygraph.estimators import DegMat

from uncertainty_estimation.deg_mat import DegMat
from uncertainty_estimation.deberta import Deberta

# Configure logging for this module.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("uncertainty_estimator")

def compute_uncertainty(generated_responses, device="cpu", batch_size=10, affinity="entail", verbose=False):
    """
    Computes the uncertainty for a list of generated responses using the DegSem (DegMat) method.
    
    Parameters:
      generated_responses (list of str): The outputs from your generative model.
      device (str): "cpu" or "cuda".
      batch_size (int): Batch size for the NLI model.
      affinity (str): "entail" (or "contra") determines how similarity is computed.
      verbose (bool): If True, logs additional debug information.
      
    Returns:
      np.ndarray: The computed uncertainty scores.
    """
    # Initialize the NLI model.
    nli_model = Deberta("microsoft/deberta-large-mnli", batch_size=batch_size, device=device)
    log.info(f"Initialized NLI model on {device} with batch_size={batch_size}")
    
    # Create the DegMat estimator.
    estimator = DegMat(nli_model, affinity=affinity, verbose=verbose)
    log.info(f"Using DegMat estimator with affinity='{affinity}'")
    
    # Compute uncertainty. The estimator expects a dictionary of statistics; here, we assume it can work directly
    # on the list of responses by iterating over them (as in the __call__ implementation).
    uncertainty_scores = estimator({"sample_texts": generated_responses})
    return uncertainty_scores

if __name__ == "__main__":
    # Test the uncertainty estimator with a set of dummy responses.
    responses = [
        "The Bayesian workflow involves data cleaning, model building, inference, and evaluation.",
        "cats and dogs are animals that can live in houses.",
        "it was on christmas eve that jesus christ was born",
        "Would you like a flag-based setup now, where you can toggle between ChatUI and Hugging Face, so you can plug in whitebox UQ later when it becomes available?"
    ]
    
    uncertainty = compute_uncertainty(responses, device="cpu", batch_size=4, affinity="entail", verbose=True)
    print("Computed uncertainty scores:", uncertainty)
