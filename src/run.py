# imports
import os
import pandas as pd
from models.data_split import get_data_split
from src.models.train import (build_param_grid,
                              build_model,
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
if __name__ == "__main__":
    # read df
    data = pd.read_csv('src/dataset/fixed_dataset.csv')

    # split data
    X_train, X_test, y_train, y_test = get_data_split(data, seed=42)

    param_grid_dt = build_param_grid(model= "decision_tree")
    param_grid_rf = build_param_grid(model= "random_forest")
    param_grid_knn = build_param_grid(model= "knn")
    param_grid_nb = build_param_grid(model= "naive_bayes")
    param_grid_xgb = build_param_grid(model= "xgboost")

    # training process
    # decision tree
    results_dt = run_pipeline(X_train=X_train,
                              y_train=y_train,
                              param_grid= param_grid_dt,
                              cv_inner= 5,
                              n_splits=5,
                              n_repeats=10,
                              output_dir="src/outputs",
                              model_filename="decision_tree_model.joblib",
                              metrics_filename="decision_tree_cv_metrics.csv",
                              random_state = 42,
                              )

    # random forest
    results_rf = run_pipeline(X_train=X_train,
                              y_train=y_train,
                              param_grid=param_grid_rf,
                              cv_inner=5,
                              n_splits=5,
                              n_repeats=10,
                              output_dir="src/outputs",
                              model_filename="random_forest_model.joblib",
                              metrics_filename="random_forest_cv_metrics.csv",
                              random_state=42,
                              )

    # knn
    results_knn = run_pipeline(X_train=X_train,
                               y_train=y_train,
                               param_grid=param_grid_knn,
                               cv_inner=5,
                               n_splits=5,
                               n_repeats=10,
                               output_dir="src/outputs",
                               model_filename="knn_model.joblib",
                               metrics_filename="knn_cv_metrics.csv",
                               random_state=42,
                               )

    # naive bayes
    results_nb = run_pipeline(X_train=X_train,
                              y_train=y_train,
                              param_grid=param_grid_nb,
                              cv_inner=5,
                              n_splits=5,
                              n_repeats=10,
                              output_dir="src/outputs",
                              model_filename="naive_bayes_model.joblib",
                              metrics_filename="naive_bayes_cv_metrics.csv",
                              random_state=42,
                              )

    # xgboost
    results_xgb = run_pipeline(X_train=X_train,
                               y_train=y_train,
                               param_grid=param_grid_dt,
                               cv_inner=5,
                               n_splits=5,
                               n_repeats=10,
                               output_dir="src/outputs",
                               model_filename="xgboost_model.joblib",
                               metrics_filename="xgboost_cv_metrics.csv",
                               random_state=42,
                               )

    # Evaluate on held-out test set,
    # only after model tuning
    # fitted_model = results["model"]
    # preds = predict_new_data(fitted_model, X_test, proba=True)
    # print("Test predictions (first 10):", preds["predictions"][:10])
    # print("Test accuracy:", accuracy_score(y_test, preds["predictions"]).round(4))

    # ── Later: load and reuse ──────────────────
    # model = load_model("outputs/decision_tree_model.joblib")
    # preds = predict_new_data(model, X_new)