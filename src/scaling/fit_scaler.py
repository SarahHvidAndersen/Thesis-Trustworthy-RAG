#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

# lex isn't working in this script

UE_COLUMNS = ['lex_score', 'deg_score', 'ecc_score']
SCALER_DIR = 'fitted_scalers'
CALIBRATION_PLOT = 'output/quantile_calibration_plot.png'
OUTPUT_CSV = 'output/large_quantile_confidences.csv'
INPUT_CSV = 'output/response_test_data/filt_anno_split_testset_merged.csv' #lex_score

# Load UE scores
def load_data(path=INPUT_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return pd.read_csv(path)

# Split into calibration / test
def split_data(df, test_size=0.2, random_state=42):
    calib, test = train_test_split(df, test_size=test_size, random_state=random_state)
    return calib, test

# Fit & save one QuantileTransformer per UE metric
def fit_and_save_scalers(calib_df, columns=UE_COLUMNS, scaler_dir=SCALER_DIR):
    os.makedirs(scaler_dir, exist_ok=True)
    scalers = {}
    for col in columns:
        qt = QuantileTransformer(
            output_distribution='uniform',
            n_quantiles=min(1000, len(calib_df)),
            subsample=len(calib_df),
            random_state=42
        )
        qt.fit(calib_df[[col]])

        print(scaler_dir)
        dump(qt, os.path.join(scaler_dir, f'{col}_quantile_scaler.joblib'))
        scalers[col] = qt
    return scalers

# apply to full df
def apply_scalers(df, scalers, columns=UE_COLUMNS):
    """
    Given a dataframe and a dict of fitted QuantileTransformers,
    populate df[f'{col}_conf_q'] = 1 - qt.transform(raw_u)
    so that low uncertainty → high confidence.
    """
    for col, qt in scalers.items():
        # raw = df[[col]]  # shape (N,1)
        quantiles = qt.transform(df[[col]]).ravel()   # shape (N,)
        df[f'{col}_conf_q'] = 1.0 - quantiles


# Plot calibration CDFs on test set 
def plot_calibration(df, test_df, columns=UE_COLUMNS, save_path=CALIBRATION_PLOT):
    plt.figure(figsize=(6,6))
    x = np.linspace(0,1,100)
    for col in columns:
        confidences = df.loc[test_df.index, f'{col}_conf_q']
        y = [np.mean(confidences <= xi) for xi in x]
        plt.plot(x, y, label=col)
    plt.plot([0,1], [0,1], 'k--', label='ideal')
    plt.xlabel('Confidence threshold $t$')
    plt.ylabel('Empirical CDF')
    plt.title('Quantile Calibration Plot')
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

# save
def save_results(df, path=OUTPUT_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

# Example usage & testing 
def main():
    # Load and split
    df = load_data()
    calib_df, test_df = split_data(df)

    # Fit scalers and apply
    scalers = fit_and_save_scalers(calib_df)
    apply_scalers(df, scalers)

    # Plot and save
    plot_calibration(df, test_df)
    save_results(df)

    # Quick sanity check: print a few before/after pairs
    # a lower raw score = higher confidence
    sample = df.sample(5)#, random_state=0)
    for col in UE_COLUMNS:
        raw = sample[col].values
        conf = sample[f'{col}_conf_q'].values
        print(f"{col:10s}  raw → conf_q")
        print(np.vstack([raw, conf]).T)
        print()

if __name__ == "__main__":
    main()
