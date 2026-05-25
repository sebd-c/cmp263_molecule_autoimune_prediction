# imports
import os
import pandas as pd
from src.models.data_split import get_data_split
from src.models.train import (build_param_grid,
                              build_model,
                              get_select_k_best_features,
                              run_cross_validation,
                              save_model
                              )
from src.plotters.get_plots import (plot_cv_metrics,
                                    plot_feature_summary_heatmap,
                                    plot_select_k_best_scores)
#################################################################################
# path global vars
INPUT_PATH = '/home/debs/python_projects/cmp263---autoimune_dataset/src/dataset/fixed_dataset.csv'
OUTPUT_DIR = '/home/debs/python_projects/cmp263---autoimune_dataset/outputs'

#################################################################################
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
                 output_dir: str = OUTPUT_DIR,
                 model_prefix: str = "decision_tree",
                 plot_format: str = "png",
                 # misc
                 random_state: int = 42,
                 ) -> dict:
    """
    Whole pipeline function from hyperparameter tuning
    to model fitting in whole dataset
    """

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
    # save it as .csv
    metrics_filename = model_prefix + "_cv_metrics.csv"
    metrics_output_path = os.path.join(output_dir, metrics_filename)
    scores_df.to_csv(metrics_output_path, index=False)
    print(scores_df.describe().T[["mean", "std"]].round(4))

    print("\n══ Step 3 / 4 — Plotting CV metrics ══")
    plot_cv_metrics(scores_df,
                    output_dir=output_dir,
                    file_format=plot_format,
                    model_prefix=model_prefix
                    )

    print("\n══ Step 4 / 4 — Fitting on full data & saving model ══")
    model_filename = model_prefix + "_model.joblib"
    model_path   = os.path.join(output_dir, model_filename)
    fitted_model = save_model(model, X_train, y_train, model_path=model_path)
    print(f"\nBest params found:\n{fitted_model.best_params_}\n")

    selected_features = get_select_k_best_features(fitted_model=fitted_model,
                                                   feature_names=X_train.columns)
    selected_features_filename = model_prefix + "_selected_features.csv"
    selected_features_path = os.path.join(output_dir, selected_features_filename)
    selected_features.to_csv(selected_features_path, index=False)
    print(f"\nSelected features saved into {selected_features_path}")
    print(selected_features[["feature", "score", "rank"]].to_string(index=False))

    plot_select_k_best_scores(features_df=selected_features,
                              output_dir=output_dir,
                              file_format=plot_format,
                              model_prefix=model_prefix,
                              )

    print("\n══ Pipeline complete ══\n")
    return {"model": fitted_model,
            "cv_scores": scores_df,
            "selected_features": selected_features,
            "model_path": model_path,
            }


####################################################
if __name__ == "__main__":
    # read df
    data = pd.read_csv(INPUT_PATH)

    # split data
    X_train, X_test, y_train, y_test = get_data_split(data, seed=42)

    # save used splits
    X_train_filename = 'x_train_data.csv'
    X_train_saved_path = os.path.join(OUTPUT_DIR, X_train_filename)
    X_train.to_csv(X_train_saved_path, index=False)

    X_test_filename = 'X_test_data.csv'
    X_test_saved_path = os.path.join(OUTPUT_DIR, X_test_filename)
    X_test.to_csv(X_test_saved_path, index=False)

    y_train_filename = 'y_train_data.csv'
    y_train_saved_path = os.path.join(OUTPUT_DIR, y_train_filename)
    y_train.to_csv(y_train_saved_path, index=False)

    y_test_filename = 'y_test_data.csv'
    y_test_saved_path = os.path.join(OUTPUT_DIR, y_test_filename)
    y_test.to_csv(y_test_saved_path, index=False)

    # make param grids by models
    param_grid_dt = build_param_grid(model= "decision_tree")
    param_grid_rf = build_param_grid(model= "random_forest")
    param_grid_knn = build_param_grid(model= "knn")
    param_grid_nb = build_param_grid(model= "naive_bayes")
    param_grid_xgb = build_param_grid(model= "xgboost")

    # plot heatmatp feats
    plot_feature_summary_heatmap(df= X_train,
                                 output_dir=OUTPUT_DIR,
                                 file_format='png')
    # training process
    # decision tree
    results_dt = run_pipeline(X_train=X_train,
                              y_train=y_train,
                              param_grid= param_grid_dt,
                              cv_inner= 5,
                              n_splits=5,
                              n_repeats=10,
                              output_dir=OUTPUT_DIR,
                              model_prefix="decision_tree",
                              random_state = 42,
                              )

    # random forest
    results_rf = run_pipeline(X_train=X_train,
                              y_train=y_train,
                              param_grid=param_grid_rf,
                              cv_inner=5,
                              n_splits=5,
                              n_repeats=10,
                              output_dir=OUTPUT_DIR,
                              model_prefix="random_forest",
                              random_state=42,
                              )

    # knn
    results_knn = run_pipeline(X_train=X_train,
                               y_train=y_train,
                               param_grid=param_grid_knn,
                               cv_inner=5,
                               n_splits=5,
                               n_repeats=10,
                               output_dir=OUTPUT_DIR,
                               model_prefix="knn_",
                               random_state=42,
                               )

    # naive bayes
    results_nb = run_pipeline(X_train=X_train,
                              y_train=y_train,
                              param_grid=param_grid_nb,
                              cv_inner=5,
                              n_splits=5,
                              n_repeats=10,
                              output_dir=OUTPUT_DIR,
                              model_prefix="naive_bayes",
                              random_state=42,
                              )

    # xgboost
    results_xgb = run_pipeline(X_train=X_train,
                               y_train=y_train,
                               param_grid=param_grid_dt,
                               cv_inner=5,
                               n_splits=5,
                               n_repeats=10,
                               output_dir=OUTPUT_DIR,
                               model_prefix="xgboost",
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
