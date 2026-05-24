# imports
import os
import pandas as pd

from src.models.train import (build_model,
                              run_cross_validation,
                              save_model
                              )
from src.plotters.get_plots import plot_cv_metrics

####################################################################

def run_pipeline(X_train,
                 y_train,
                 # search config
                 param_grid: dict | None = None,
                 search_scoring: str = "f1_weighted",
                 cv_inner: int = 5,
                 # outer CV config
                 cv_scoring: dict | None = None,
                 n_splits: int = 5,
                 n_repeats: int = 10,
                 # output paths
                 output_dir: str = "outputs",
                 model_filename: str = "decision_tree_model.joblib",
                 metrics_filename: str = "cv_metrics.csv",
                 plot_format: str = "png",
                 # misc
                 random_state: int = 42,
                 ) -> dict:
    """
    Whole pipeline function from hyperparameter tuning
    to model fitting in whole dataset
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n══ Step 1 / 4 — Building GridSearchCV model ══")
    model = build_model(param_grid= param_grid,
                        scoring= search_scoring,
                        cv_inner= cv_inner,
                        random_state = random_state,
                        )

    print("\n══ Step 2 / 4 — Running outer cross-validation ══")
    scores_df = run_cross_validation(model= model,
                                     X= X_train,
                                     y= y_train,
                                     n_splits= n_splits,
                                     n_repeats= n_repeats,
                                     random_state= random_state,
                                     scoring= cv_scoring,
                                     )
    print(scores_df.describe().T[["mean", "std"]].round(4))

    print("\n══ Step 3 / 4 — Plotting CV metrics ══")
    plots_dir = os.path.join(output_dir, "plots")
    plot_cv_metrics(scores_df, output_dir=plots_dir, file_format=plot_format)

    print("\n══ Step 4 / 4 — Fitting on full data & saving model ══")
    model_path   = os.path.join(output_dir, model_filename)
    fitted_model = save_model(model, X_train, y_train, model_path=model_path)
    print(f"\nBest params found:\n{fitted_model.best_params_}\n")

    print("\n══ Pipeline complete ══\n")
    return {"model": fitted_model,
            "cv_scores": scores_df,
            "model_path": model_path,
            }

####################################################
#TODO: fix main
if __name__ == "__main__":
    pass