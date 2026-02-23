import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error


def evaluate_model(model, X_test, y_test):
    """
    Returns Pearson r and MSE on test set.
    """
    y_pred = model.predict(X_test)

    r = pearsonr(y_test, y_pred)[0]
    mse = mean_squared_error(y_test, y_pred)

    return r, mse


def save_results(results_dict, output_path="results/final_results.csv"):
    """
    Saves results dictionary to CSV.
    """
    df = pd.DataFrame(results_dict)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
