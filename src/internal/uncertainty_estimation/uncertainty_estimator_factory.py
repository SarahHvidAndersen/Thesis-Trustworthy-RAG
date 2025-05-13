
def get_uncertainty_estimator(method: str, **kwargs):
    """
    Returns an instantiated uncertainty estimator based on the specified method.
    
    The kwargs should be provided from a structured configuration (such as a YAML file)
    so that only the parameters relevant to the chosen method are passed.
    
    Parameters:
      method (str): one of "lexical_similarity", "deg_mat", or "eccentricity".
      kwargs: should contain all necessary parameters for the chosen method.
          For example, for "deg_mat", kwargs must include:
             - batch_size (int)
             - device (str)
             - affinity (str)
             - verbose (bool)
          For "lexical_similarity", kwargs might include:
             - metric (str)
          For "eccentricity", kwargs might include:
             - similarity_score (str)
             - affinity (str)
             - verbose (bool)
             - thres (float)
    
    Returns:
      An instance of the uncertainty estimator.
    """
    method = method.lower()
    if method == "lexical_similarity":
        from internal.uncertainty_estimation.lexical_similarity import LexicalSimilarity
        return LexicalSimilarity(**kwargs)
    elif method == "deg_mat":
        from internal.uncertainty_estimation.deg_mat import DegMat
        from internal.uncertainty_estimation.deberta import Deberta
        # Expect the config to provide these parameters.
        batch_size = kwargs.pop("batch_size")
        device = kwargs.pop("device")
        affinity = kwargs.pop("affinity")
        verbose = kwargs.pop("verbose")
        # Initialize the NLI model.
        nli_model = Deberta("microsoft/deberta-large-mnli", batch_size=batch_size, device=device)
        return DegMat(nli_model, affinity=affinity, verbose=verbose, **kwargs)
    elif method == "eccentricity":
        from internal.uncertainty_estimation.eccentricity import Eccentricity
        return Eccentricity(**kwargs)
    else:
        raise ValueError(f"Unknown uncertainty method: {method}")



def compute_uncertainty(estimator, samples: list):
    """
    Computes the uncertainty score given an estimator and a list of generated samples.
    For estimators that require additional pre-processing (like Eccentricity), this function
    takes care of building the proper stats dictionary.
    
    Parameters:
      estimator: an uncertainty estimator instance.
      samples: a list of generated text samples.
    
    Returns:
      A float uncertainty score.
    """
    # For eccentricity, we need the semantic matrix.
    # import compute_sim_score from the local common module.
    from internal.uncertainty_estimation.common import compute_sim_score
    import numpy as np
    # detect eccentricity by checking its class name.
    if estimator.__class__.__name__.lower() == "eccentricity":
        W = compute_sim_score(answers=samples, 
                              affinity=estimator.affinity, 
                              similarity_score=estimator.similarity_score)
        stats = {
            "sample_texts": [samples],               # Expected shape: (1, n)
            "semantic_matrix_entail": np.array([W])    # Expected shape: (1, n, n)
        }
    else:
        # For other estimators, only the sample_texts key is needed.
        stats = {"sample_texts": [samples]}
    
    uncertainty_array = estimator(stats)
    print(uncertainty_array)
    # Assuming the estimator returns a 1-D np.array, take the first value.
    return float(uncertainty_array[0])
