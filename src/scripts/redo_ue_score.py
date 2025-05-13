import ast, json
import pandas as pd
from internal.uncertainty_estimation.uncertainty_estimator_factory import (
    get_uncertainty_estimator, compute_uncertainty)

INPUT  = "output/answered_test_data/testset_with_predictions.csv"
OUTPUT = INPUT                  # overwrite

#  utilities 
def to_list(cell):
    if isinstance(cell, list):   return cell
    if isinstance(cell, str):
        try:    return json.loads(cell)
        except json.JSONDecodeError:
            return ast.literal_eval(cell)
    raise TypeError(f"'samples' must be str or list, got {type(cell)}")

#  load
df = pd.read_csv(INPUT)
df["samples_parsed"] = df["samples"].apply(to_list)

# build estimators
lex_est = get_uncertainty_estimator("lexical_similarity")
ecc_est = get_uncertainty_estimator(
    "eccentricity",
    thres=0.7
)

# compute scores again
df["lex_score"] = df["samples_parsed"].apply(
    lambda s: compute_uncertainty(lex_est, s))

df["ecc_score"] = df["samples_parsed"].apply(
    lambda s: compute_uncertainty(ecc_est, s))

# save & report 
df.drop(columns="samples_parsed").to_csv(OUTPUT, index=False)
print(f"Saved updated lex_score & ecc_score to {OUTPUT}")
