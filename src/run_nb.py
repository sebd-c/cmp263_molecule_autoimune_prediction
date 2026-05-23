# imports
import os
import pandas as pd
from typing import Optional
from models.naive_bayes import (build_model, run_cross_validation, save_model)
from plotters.get_plots import plot_cv_metrics
from pre_processing.input_fix import NumericalDiscretizer
from pre_processing import dataset_fix
from models.data_split import get_data_split

####################################################################

def run_nb_pipeline(df_raw,
                    # discretization config
                    method: str = 'quantile',
                    n_bins: int = 4,
                    # search config
                    param_grid: Optional[dict] = None,
                    search_scoring: str = "f1_weighted",
                    cv_inner: int = 5,
                    # outer CV config
                    cv_scoring: Optional[dict] = None,
                    n_splits: int = 5,
                    n_repeats: int = 10,
                    # output paths
                    output_dir: str = "outputs",
                    model_filename: str = "naive_bayes_model.joblib",
                    discretizer_filename: str = "discretizer.joblib",
                    metrics_filename: str = "nb_cv_metrics.csv",
                    plot_format: str = "png",
                    # misc
                    seed: int = 42,
                    ) -> dict:
    """
    Full pipeline for Naive Bayes WITHOUT data leakage.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n══ Step 1 / 7 — Getting variable types ══")
    vartypes_dic = dataset_fix.get_vartypes(df_raw)
    
    print("\n══ Step 2 / 7 — Splitting data (stratified) ══")
    X_train_df, X_test_df, y_train, y_test = get_data_split(df_raw, seed=seed)
    
    print("\n══ Step 3 / 7 — Fitting discretizer on TRAINING data only ══")
    discretizer = NumericalDiscretizer(vartypes_dic, method, n_bins)
    X_train_binned = discretizer.fit_transform(X_train_df)
    
    # Select only binary columns + binned columns for training
    binary_columns = [col for col in X_train_df.columns if vartypes_dic.get(col) == 'binary']
    bin_columns = [col for col in X_train_binned.columns if col.endswith('_bin')]
    X_train_nb = pd.concat([X_train_df[binary_columns], X_train_binned[bin_columns]], axis=1)
    
    print(f"Training set shape: {X_train_nb.shape}")
    print(f"Number of features: {X_train_nb.shape[1]}")
    
    print("\n══ Step 4 / 7 — Building GridSearchCV model ══")
    model = build_model(param_grid=param_grid,
                        scoring=search_scoring,
                        cv_inner=cv_inner,
                        random_state=seed,
                        verbose=1,
                        )

    print("\n══ Step 5 / 7 — Running cross-validation ══")
    
    # Define scoring metrics
    if cv_scoring is None:
        cv_scoring = {
            'accuracy': 'accuracy',
            'f1_weighted': 'f1_weighted',
            'precision_weighted': 'precision_weighted',
            'recall_weighted': 'recall_weighted'
        }
    
    scores_df = run_cross_validation(model=model,
                                     X=X_train_nb,
                                     y=y_train,
                                     n_splits=n_splits,
                                     n_repeats=n_repeats,
                                     random_state=seed,
                                     scoring=cv_scoring,
                                     )
    print("\nCross-validation results:")
    print(scores_df.describe().T[["mean", "std"]].round(4))

    print("\n══ Step 6 / 7 — Plotting CV metrics ══")
    #plot now working yet
    #plot_cv_metrics(scores_df)

    print("\n══ Step 7 / 7 — Fitting on full training data & saving ══")
    model_path = os.path.join(output_dir, model_filename)
    discretizer_path = os.path.join(output_dir, discretizer_filename)
    fitted_model = save_model(model, discretizer, X_train_nb, y_train, 
                              model_path, discretizer_path)
    print(f"\nBest params found:\n{fitted_model.best_params_}\n")

    # Transform test set using fitted discretizer (NO LEAKAGE)
    print("\n══ Optional: Transforming test set ══")
    X_test_binned = discretizer.transform(X_test_df)
    X_test_nb = pd.concat([X_test_df[binary_columns], X_test_binned[bin_columns]], axis=1)
    print(f"Test set shape: {X_test_nb.shape}")

    print("\n══ Pipeline complete ══\n")
    
    return {"model": fitted_model,
            "cv_scores": scores_df,
            "model_path": model_path,
            "discretizer_path": discretizer_path,
            "X_train": X_train_nb,
            "y_train": y_train,
            "X_test": X_test_nb,
            "y_test": y_test,
            "vartypes_dic": vartypes_dic,
            }


####################################################
if __name__ == "__main__":
    
    path_to_og_dataset_test = "src/dataset/DIA_testset_RDKit_descriptors.csv"
    path_to_og_dataset_train = "src/dataset/DIA_trainingset_RDKit_descriptors.csv"
    
    df_og = dataset_fix.unite_dataset(path_to_og_dataset_test, path_to_og_dataset_train)
    df_fix, dic_fix_vartypes, empty_instances, list_clones, list_homo = (
            dataset_fix.fix_dataset(df_og, remove_smiles=True)
    )
    
    print(f"Dataset shape after fixing: {df_fix.shape}")
    print(f"Empty instances found: {empty_instances}")
    print(f"Clones found: {len(list_clones)}")
    print(f"Homogenous columns removed: {len(list_homo)}")
        
    results = run_nb_pipeline(df_fix, output_dir="outputs", n_bins=4)
        
    print(f"Model saved at: {results['model_path']}")
    print(f"Discretizer saved at: {results['discretizer_path']}")
    print(f"Best parameters: {results['model'].best_params_}")