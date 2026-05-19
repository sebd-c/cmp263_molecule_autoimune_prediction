
# imports
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import (GridSearchCV,
                                     RepeatedStratifiedKFold,
                                     cross_validate,
                                    )

#####################################################################
# module with functions of training process
def build_param_grid() -> dict:
    """
    Return the hyperparameter grid for GridSearchCV.
    """

    # define model parameters of interest
    return {
        "criterion":         ["gini", "entropy"],
        "max_depth":         [None, 3, 5, 10, 20],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf":  [1, 2, 5, 10],
        "max_features":      [None, "sqrt", "log2"],
        "class_weight":      ["balanced"],
    }


# First repetition of the cross-validation nest
def build_model(param_grid: dict | None = None,
                scoring: str = "f1_weighted",
                cv_inner: int = 5,
                random_state: int = 42,
                verbose: int = 3,
                ) -> GridSearchCV:
    """
    Wrapper of a DecisionTreeClassifier in a GridSearchCV.
    """

    # define classifier to be used
    base_estimator = DecisionTreeClassifier(random_state=random_state)

    # start search
    search = GridSearchCV(estimator= base_estimator,
                          param_grid= param_grid,
                          scoring= scoring,
                          cv= cv_inner,
                          refit= True,
                          verbose= verbose,
    )
    return search


# Second repetition of the cross-validation nest
def run_cross_validation(model,
                         X,
                         y,
                         n_splits: int = 5,
                         n_repeats: int = 10,
                         random_state: int = 42,
                         scoring: dict | None = None,
                        ) -> pd.DataFrame:
    """
    Evaluate *model* with RepeatedStratifiedKFold and return a tidy DataFrame
    of per-fold scores.
    """

    # define cross validation settings
    cv = RepeatedStratifiedKFold(n_splits= n_splits,
                                 n_repeats= n_repeats,
                                 random_state= random_state,
                                )

    # run cross-validation with specified model
    cv_results = cross_validate(model,
                                X,
                                y,
                                cv= cv,
                                scoring= scoring,
                                return_train_score = True,
                                )

    # Keep only test scores and strip the "test_" prefix
    scores_df = pd.DataFrame(cv_results)

    return scores_df


# post model training functions, i.e. run on test dataset, save model...
def save_model(model,
               X_train,
               y_train,
               model_path: str,
              ):
    """
    Fit *model* on the full training set and saves it as joblib.
    """
    print("[…] Fitting model on full training data…")
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print(f"Fitted model saved into {model_path}")
    return model

def load_model(model_path: str):
    """
    Load and return a previously saved model.
    """
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model
