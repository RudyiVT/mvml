import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from feature_engine.encoding import RareLabelEncoder

def clean_text(text: str) -> str:
    """Basic text cleaning."""

    return (
        text.lower()
        .replace("[!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~]", "")
        .replace("\n", "")
        .replace("[^\w\s]", "")
    )


def define_model():
    """
    Define a pipeline for data preprocessing and modelling

    Steps:
        1. Pipeline for text features
        2. Pipeline for categorical features
        3. Pipeline for numerical measurements of text data
        4. Assign specific transformations for each column
        5. Combine data preprocessing and modelling into one pipeline

    Returns:

    """

    # Pipeline for text features
    text_transformer = Pipeline(
        steps=[
            ("text_imputer", SimpleImputer(strategy="constant", fill_value="")),
            ("reshape", FunctionTransformer(np.reshape, kw_args={"newshape": -1})),
            (
                "tfidf_encoder",
                TfidfVectorizer(
                    lowercase=True,
                    preprocessor=clean_text,
                    stop_words="english",
                    max_df=0.1,
                    min_df=0.01,
                    ngram_range=(1, 3),
                ),
            ),
        ],
    )

    # Pipeline for categorical features
    categorical_transformer = Pipeline(
        steps=[
            ("text_imputer", SimpleImputer(strategy="constant", fill_value="")),
            ("rare_encoder", RareLabelEncoder(tol=0.001)),
            ("oh_encoder", OneHotEncoder(handle_unknown="ignore")),
        ],
    )

    # Pipeline for numerical measurements of text data
    text_overall_transformer = Pipeline(
        steps=[
            ("text_imputer", SimpleImputer(strategy="constant", fill_value="")),
            ("text_len", FunctionTransformer(func=np.vectorize(len))),
        ],
    )

    # Assign specific transformations for each column
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_transformer, ["text"]),
            ("title", text_transformer, ["title"]),
            ("author", categorical_transformer, ["author"]),
            ("text_overall", text_overall_transformer, ["text", "title", "author"]),
        ],
        n_jobs=-1,
    )

    # Combine data preprocessing and modelling into one pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500, n_jobs=-1, verbose=0, class_weight="balanced"
                ),
            ),
        ],
        verbose=True,
    )

    return model


def fit_model(model, data: pd.DataFrame, param_grid: dict) -> None:
    # Define params search grid
    grid = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=5,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=True,
    )
    grid.fit(data, data.label)

    return grid
