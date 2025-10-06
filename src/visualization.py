
"""
visualization.py
----------------
Lightweight plotting helpers using matplotlib only.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def plot_log_saleprice_hist(train: pd.DataFrame, out_path: str = "results/hist_log_saleprice.png") -> str:
    if "SalePrice" not in train:
        raise KeyError("SalePrice column not found.")
    _ensure_dir(out_path)
    values = np.log1p(train["SalePrice"])
    plt.figure()
    plt.hist(values, bins=40)
    plt.title("Distribution of log1p(SalePrice)")
    plt.xlabel("log1p(SalePrice)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path

def plot_scatter(train: pd.DataFrame, feature: str, out_path: str = None) -> str:
    if "SalePrice" not in train or feature not in train.columns:
        raise KeyError("Required columns missing.")
    if out_path is None:
        out_path = f"results/scatter_{feature}_SalePrice.png"
    _ensure_dir(out_path)
    plt.figure()
    plt.scatter(train[feature], train["SalePrice"], s=12, alpha=0.8)
    plt.title(f"{feature} vs SalePrice")
    plt.xlabel(feature)
    plt.ylabel("SalePrice")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path
