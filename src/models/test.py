import argparse
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score,
                             balanced_accuracy_score,
                             classification_report,
                             confusion_matrix,
                             f1_score,
                             precision_score,
                             recall_score,
                             roc_auc_score)


DEFAULT_OUTPUT_DIR = "/home/debs/python_projects/cmp263---autoimune_dataset/outputs"
DEFAULT_TARGET_COLUMN = "Label"


def load_model(model_path: str):
    """
    Load a fitted sklearn-compatible model saved as a .joblib file.
    """
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model


def load_unseen_data(data_path: str,
                     target_column: str = DEFAULT_TARGET_COLUMN,
                     y_path: str | None = None,
                     ) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Load unseen data for testing.

    If y_path is provided, data_path is treated as X and y_path as labels.
    Otherwise, data_path may contain target_column; when present it is split
    into X and y. If no labels are available, y is returned as None.
    """
    X = pd.read_csv(data_path)

    if y_path is not None:
        y = _read_target(y_path, target_column=target_column)
        return X, y

    if target_column in X.columns:
        y = X[target_column]
        X = X.drop(columns=[target_column])
        return X, y

    return X, None


def align_features_to_model(model,
                            X: pd.DataFrame,
                            ) -> pd.DataFrame:
    """
    Reorder unseen data columns to match the fitted model when possible.

    SearchCV objects expose the fitted pipeline through best_estimator_.
    Pipelines fitted on pandas DataFrames usually carry feature_names_in_.
    """
    estimator = getattr(model, "best_estimator_", model)
    expected_features = getattr(estimator, "feature_names_in_", None)

    if expected_features is None:
        return X

    expected_features = list(expected_features)
    missing_features = [col for col in expected_features if col not in X.columns]

    if missing_features:
        raise ValueError("Unseen data is missing model features: "
                         f"{missing_features}")

    return X.loc[:, expected_features]


def predict_unseen_data(model,
                        X: pd.DataFrame,
                        positive_class: int = 1,
                        ) -> pd.DataFrame:
    """
    Predict classes and, when available, class probabilities.
    """
    X_model = align_features_to_model(model, X)
    predictions = model.predict(X_model)

    pred_df = pd.DataFrame({"prediction": predictions})

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_model)
        classes = list(getattr(model, "classes_", []))

        if positive_class in classes:
            positive_class_idx = classes.index(positive_class)
        elif probabilities.shape[1] == 2:
            positive_class_idx = 1
        else:
            positive_class_idx = None

        if positive_class_idx is not None:
            pred_df["positive_probability"] = probabilities[:, positive_class_idx]

    return pred_df


def evaluate_predictions(y_true,
                         predictions: pd.DataFrame,
                         positive_class: int = 1,
                         zero_division: int = 0,
                         ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute final classification metrics, classification report, and
    confusion matrix for an unseen labeled dataset.
    """
    y_pred = predictions["prediction"]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred,
                                               average="weighted",
                                               zero_division=zero_division),
        "recall_weighted": recall_score(y_true, y_pred,
                                        average="weighted",
                                        zero_division=zero_division),
        "f1_weighted": f1_score(y_true, y_pred,
                                average="weighted",
                                zero_division=zero_division),
        "precision_macro": precision_score(y_true, y_pred,
                                           average="macro",
                                           zero_division=zero_division),
        "recall_macro": recall_score(y_true, y_pred,
                                     average="macro",
                                     zero_division=zero_division),
        "f1_macro": f1_score(y_true, y_pred,
                             average="macro",
                             zero_division=zero_division),
    }

    if "positive_probability" in predictions.columns:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true,
                                               predictions["positive_probability"])
        except ValueError:
            metrics["roc_auc"] = None

    metrics_df = pd.DataFrame([metrics])
    report_df = pd.DataFrame(classification_report(y_true, y_pred,
                                                   output_dict=True,
                                                   zero_division=zero_division)).T
    labels = sorted(pd.Series(y_true).dropna().unique())
    cm_df = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels),
                         index=[f"true_{label}" for label in labels],
                         columns=[f"pred_{label}" for label in labels])

    return metrics_df, report_df, cm_df


def run_test_pipeline(model_path: str,
                      data_path: str,
                      output_dir: str = DEFAULT_OUTPUT_DIR,
                      target_column: str = DEFAULT_TARGET_COLUMN,
                      y_path: str | None = None,
                      model_prefix: str | None = None,
                      positive_class: int = 1,
                      ) -> dict:
    """
    Load a .joblib model, predict an unseen dataset, and save final scores.

    Saves:
    - <model_prefix>_test_predictions.csv
    - <model_prefix>_test_metrics.csv, when labels are available
    - <model_prefix>_classification_report.csv, when labels are available
    - <model_prefix>_confusion_matrix.csv, when labels are available
    """
    os.makedirs(output_dir, exist_ok=True)

    model = load_model(model_path)
    X_unseen, y_unseen = load_unseen_data(data_path=data_path,
                                          target_column=target_column,
                                          y_path=y_path)

    predictions = predict_unseen_data(model=model,
                                      X=X_unseen,
                                      positive_class=positive_class)

    if y_unseen is not None:
        predictions.insert(0, "true_label", y_unseen.reset_index(drop=True))

    if model_prefix is None:
        model_prefix = Path(model_path).stem.replace("_model", "")

    predictions_path = os.path.join(output_dir,
                                    f"{model_prefix}_test_predictions.csv")
    predictions.to_csv(predictions_path, index=False)
    print(f"Test predictions saved @ {predictions_path}")

    results = {"model": model,
               "predictions": predictions,
               "predictions_path": predictions_path}

    if y_unseen is None:
        print("No labels found. Metrics were not computed.")
        return results

    metrics_df, report_df, cm_df = evaluate_predictions(
        y_true=y_unseen,
        predictions=predictions,
        positive_class=positive_class,
    )

    metrics_path = os.path.join(output_dir, f"{model_prefix}_test_metrics.csv")
    report_path = os.path.join(output_dir,
                               f"{model_prefix}_classification_report.csv")
    cm_path = os.path.join(output_dir, f"{model_prefix}_confusion_matrix.csv")

    metrics_df.to_csv(metrics_path, index=False)
    report_df.to_csv(report_path)
    cm_df.to_csv(cm_path)

    print(f"Test metrics saved @ {metrics_path}")
    print(metrics_df.round(4).to_string(index=False))
    print(f"Classification report saved @ {report_path}")
    print(f"Confusion matrix saved @ {cm_path}")

    results.update({"metrics": metrics_df,
                    "classification_report": report_df,
                    "confusion_matrix": cm_df,
                    "metrics_path": metrics_path,
                    "classification_report_path": report_path,
                    "confusion_matrix_path": cm_path})

    return results


def _read_target(y_path: str,
                 target_column: str = DEFAULT_TARGET_COLUMN,
                 ) -> pd.Series:
    """
    Read labels from a one-column CSV or from a CSV containing target_column.
    """
    y_df = pd.read_csv(y_path)

    if target_column in y_df.columns:
        return y_df[target_column]

    if y_df.shape[1] == 1:
        return y_df.iloc[:, 0]

    raise ValueError(f"Could not infer target column from {y_path}. "
                     f"Expected a '{target_column}' column or one-column CSV.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved .joblib model "
                                                 "on an unseen dataset.")
    parser.add_argument("--model-path", required=True,
                        help="Path to fitted .joblib model.")
    parser.add_argument("--data-path", required=True,
                        help="Path to unseen X CSV, or CSV containing Label.")
    parser.add_argument("--y-path", default=None,
                        help="Optional path to labels CSV.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory where reports will be saved.")
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN,
                        help="Target column name when labels are in data CSV.")
    parser.add_argument("--model-prefix", default=None,
                        help="Prefix for output filenames.")
    parser.add_argument("--positive-class", type=int, default=1,
                        help="Class used for positive_probability and ROC AUC.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_test_pipeline(model_path=args.model_path,
                      data_path=args.data_path,
                      output_dir=args.output_dir,
                      target_column=args.target_column,
                      y_path=args.y_path,
                      model_prefix=args.model_prefix,
                      positive_class=args.positive_class)
