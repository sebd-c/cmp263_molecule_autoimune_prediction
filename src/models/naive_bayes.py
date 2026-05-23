import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_validate

from pre_processing.input_fix import NumericalDiscretizer


def build_param_grid() -> dict:
    return {
        "alpha": [0.1, 0.5, 1.0, 2.0, 5.0],
        "norm": [True, False],
    }


def build_model(param_grid: Optional[dict] = None,
                scoring: str = "f1_weighted",
                cv_inner: int = 5,
                random_state: int = 42,
                verbose: int = 3,
                ) -> GridSearchCV:
    base_estimator = ComplementNB()

    if param_grid is None:
        param_grid = build_param_grid()

    search = GridSearchCV(estimator=base_estimator,
                          param_grid=param_grid,
                          scoring=scoring,
                          cv=cv_inner,
                          refit=True,
                          verbose=verbose,
    )
    return search


def run_cross_validation(model,
                         X,
                         y,
                         n_splits: int = 5,
                         n_repeats: int = 10,
                         random_state: int = 42,
                         scoring: Optional[dict] = None,
                        ) -> pd.DataFrame:
    cv = RepeatedStratifiedKFold(n_splits=n_splits,
                                 n_repeats=n_repeats,
                                 random_state=random_state,
                                )

    cv_results = cross_validate(model,
                                X,
                                y,
                                cv=cv,
                                scoring=scoring,
                                return_train_score=True,
                                )

    scores_df = pd.DataFrame(cv_results)
    return scores_df


def save_model(model,
               discretizer,
               X_train,
               y_train,
               model_path: str,
               discretizer_path: str,
              ):
    """
    Fit *model* on the full training set and saves both model and discretizer.
    """
    print("[…] Fitting model on full training data…")
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    joblib.dump(discretizer, discretizer_path)
    print(f"Fitted model saved into {model_path}")
    print(f"Discretizer saved into {discretizer_path}")
    return model


def load_model(model_path: str, discretizer_path: str):
    """
    Load and return a previously saved model and discretizer.
    """
    model = joblib.load(model_path)
    discretizer = joblib.load(discretizer_path)
    print(f"Model loaded from {model_path}")
    print(f"Discretizer loaded from {discretizer_path}")
    return model, discretizer


def prepare_naive_bayes_input(df, vartypes_dic, method='quantile', n_bins=4):
    """
    Legacy function - kept for compatibility but use NumericalDiscretizer instead.
    """
    discretizer = NumericalDiscretizer(vartypes_dic, method, n_bins)
    df_binned = discretizer.fit_transform(df)
    
    binary_columns = [col for col in df.columns if vartypes_dic.get(col) == 'binary']
    bin_columns = [col for col in df_binned.columns if col.endswith('_bin')]
    
    X = pd.concat([df[binary_columns], df_binned[bin_columns]], axis=1)
    
    return X, discretizer