import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.isotonic import IsotonicRegression 
from sklearn.linear_model import LogisticRegression

GOOD_THRESHOLD = 0.70 

UE_COLUMNS = ['lex_score', 'deg_score', 'ecc_score']
SCALER_DIR = 'data/fitted_scalers'
CALIB_PLOT = 'output/quantitative_metrics/quantile_calibration_plot.png'
INPUT_CSV = 'output/quantitative_metrics/alignscore_testset_with_predictions.csv' 
OUTPUT    = 'output/quantitative_metrics/all_scalers_testset.csv'

# Load UE scores
def load_data(path=INPUT_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return pd.read_csv(path)

def add_split_column(df, test_size=0.2, random_state=42):
    """
    Add a `split` column to df in‑place, marking each row 'calib' or 'test'.
    Returns two Boolean masks for convenience.
    """
    calib_idx, test_idx = train_test_split(
        df.index, test_size=test_size, random_state=random_state, shuffle=True
    )
    df['split'] = 'calib'
    df.loc[test_idx, 'split'] = 'test'
    return df.index.isin(calib_idx), df.index.isin(test_idx)

# Split into calibration / test
def split_data(df, test_size=0.2, random_state=42):
    calib, test = train_test_split(df, test_size=test_size, random_state=random_state)
    return calib, test


def fit_and_save_scalers(df, calib_mask,
                         columns=UE_COLUMNS,
                         scaler_dir=SCALER_DIR):
    """
    Fit *both* a QuantileTransformer and an IsotonicRegression
    on the calibration split.  Saves:
        {col}_quantile_scaler.joblib
        {col}_isotonic_scaler.joblib
        {col}_sigmoid_scaler.joblib
    """
    os.makedirs(scaler_dir, exist_ok=True)
    scalers = {}

    # --- binary label for isotonic --------------------------------
    y_good = (df.loc[calib_mask, "alignscore"] >= GOOD_THRESHOLD).astype(int)

    for col in columns:
        # 1️  Quantile
        qt = QuantileTransformer(
            output_distribution='uniform',
            n_quantiles=min(1000, calib_mask.sum()),
            subsample=calib_mask.sum(),
            random_state=42,
        )
        qt.fit(df.loc[calib_mask, [col]])
        dump(qt, os.path.join(scaler_dir, f'{col}_quantile_scaler.joblib'))

        # 2️ Isotonic – note the *negative* sign so HIGH = more confident
        x_train = -df.loc[calib_mask, col].values
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        iso.fit(x_train, y_good)
        dump(iso, os.path.join(scaler_dir, f'{col}_isotonic_scaler.joblib'))

        # Sigmoid (Platt scaling) with L-BFGS
        x_train = -df.loc[calib_mask, [col]]          # keep feature name
        sig = LogisticRegression(solver="lbfgs")
        sig.fit(x_train, y_good)
        dump(sig, os.path.join(scaler_dir, f"{col}_sigmoid_scaler.joblib"))
        scalers[col] = (qt, iso, sig)   # store triple

    return scalers

# apply to full df
def apply_scalers(df, scalers, columns=UE_COLUMNS):
    """
    Adds two confidence columns per UE metric:
        *_conf_q   – 1 − quantile
        *_conf_iso – probability from isotonic regression
        *_conf_sig - sigmoid
    """
    for col, (qt, iso, sig) in scalers.items():
        # Quantile 
        df[f'{col}_conf_q'] = 1.0 - qt.transform(df[[col]]).ravel()

        # Isotonic
        df[f'{col}_conf_iso'] = iso.transform(-df[col].values)

        #sigmoid
        df[f"{col}_conf_sig"] = sig.predict_proba(df[[col]])[:,1]


# Plot calibration CDFs on test set 
def plot_calibration(df, test_mask,
                     columns=UE_COLUMNS,
                     save_path=CALIB_PLOT):
    """
    Reliability diagram for confidences on the test split.
    """
    plt.figure(figsize=(6, 6))
    x = np.linspace(0, 1, 100)
    for col in columns:
        conf = df.loc[test_mask, f'{col}_conf_q']
        y = [(conf <= t).mean() for t in x]
        plt.plot(x, y, label=f"{col} conf quantile")
    plt.plot([0, 1], [0, 1], 'k--', label='ideal')
    plt.xlabel('Predicted confidence')
    plt.ylabel('Empirical CDF')
    plt.title('Quantile Calibration – Test Split')
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

def safe_min_max(series):
    lo, hi = series.min(), series.max()
    return lo, hi if hi > lo else lo + 1e-8


def main():
    # Load and mark split
    df = load_data()
    calib_mask, test_mask = add_split_column(df)

    # Fit scalers and apply to *all* rows
    scalers = fit_and_save_scalers(df, calib_mask)
    apply_scalers(df, scalers)

    # Calibration plot on the real hold‑out
    plot_calibration(df, test_mask)

    # Save artefacts
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    df.to_csv(OUTPUT, index=False)               # full set with split

    # Quick visual sanity check
    sample = df.sample(5, random_state=0)
    for col in UE_COLUMNS:
        print(f"{col:10s}   raw → conf_q")
        for raw, conf in zip(sample[col], sample[f'{col}_conf_q']):
            print(f"{raw: .4f} → {conf:.4f}")
        print()

if __name__ == "__main__":
    main()

