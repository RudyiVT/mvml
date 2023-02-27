import joblib
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.modeling import define_model, fit_model
from src.evaluation import get_model_summary_message, get_grid_summary_message, test_data_evaluation


def run(data_path: str, model_output_path: str):
    # load data
    print("Data loading...")
    df = pd.read_csv(data_path).sample(10000)

    train_df, test_df = train_test_split(
        df, random_state=123, train_size=0.8, stratify=df.label
    )

    # Define a model and a grid search
    print("Model definition...")
    model = define_model()
    params = {
        "classifier__n_estimators": np.linspace(50, 5000, 10).astype(int),
        "classifier__max_depth": [None, 5, 7, 9],
        "preprocessor__title__tfidf_encoder__binary": [True, False],
        "preprocessor__text__tfidf_encoder__binary": [True, False],
        "preprocessor__title__tfidf_encoder__norm": [None, "l1", "l2"],
        "preprocessor__text__tfidf_encoder__norm": [None, "l1", "l2"],
    }

    # Fit the model and tune params
    print("Params tuning and evaluation...")
    model_grid = fit_model(model=model, data=train_df, param_grid=params)
    cl = model_grid.best_estimator_
    cl.fit(train_df, train_df.label)
    get_grid_summary_message(model_grid)

    # Model evaluation and test data
    print("The best model performance on the eval dataset")
    test_proba = cl.predict_proba(test_df)[:, 1]
    get_model_summary_message(test_df.label, test_proba, th=0.5)

    # Fit the final model on whole dataset
    print("Fitting the final model...")
    final_cl = define_model()
    final_cl.set_params(**model_grid.best_params_)
    final_cl.fit(df, df.label)

    # Store the model
    print("Storing the final model...")
    joblib.dump(final_cl, model_output_path)


if __name__ == "__main__":
    # parse input args
    parser = argparse.ArgumentParser(description='Fake news model training')
    parser.add_argument('--train_data_path', type=str, help='Path to a train dataset', required=True)
    parser.add_argument('--model_output_path', type=str, help='Path to store the final model', required=True)
    parser.add_argument('--test_data_path', type=str, help='Path to a test dataset', required=False)
    parser.add_argument('--test_labels_path', type=str, help='Path to a test labels', required=False)

    args = parser.parse_args()

    # run training flow
    run(data_path=args.train_data_path, model_output_path=args.model_output_path)

    # Evaluation on test data
    test_data_path = args.test_data_path
    test_labels_path = args.test_labels_path
    if test_data_path and test_labels_path:
        print("Model evalutation on a test dataset")
        test_data_evaluation(test_data_path, test_labels_path, args.model_output_path)
