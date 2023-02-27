import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    accuracy_score,
)


def get_grid_summary_message(model_grid: dict):
    print("===================================================================")
    print("The best pipeline params")
    for k, v in model_grid.best_params_.items():
        print(f"\t {k}: {v}")
    print("===================================================================")


def get_model_summary_message(y_true, y_score, th: float=0.5):
    preds = np.where(y_score > th, 1, 0)
    eval_message = f"""
===================================================================
Confusion matrix: 
{confusion_matrix(y_true, preds)}

Accuracy:      {accuracy_score(y_true, preds):.4f}
Precision:     {precision_score(y_true, preds):.4f}
Recall:        {recall_score(y_true, preds):.4f}
F1:            {f1_score(y_true, preds):.4f}
ROC AUC:       {roc_auc_score(y_true, y_score):.4f}
===================================================================
    """
    print(eval_message)


def test_data_evaluation(test_data_path: str, lables_path: str, model_path: str):
    data = pd.read_csv(test_data_path)
    labels = pd.read_csv(lables_path)
    cl = joblib.load(model_path)

    probs = cl.predict_proba(data)[:, 1]
    get_model_summary_message(labels.label, probs)
