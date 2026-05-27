import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# biblios de ferramentas de explicabilidade
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import permutation_importance
import shap
from sklearn.inspection import permutation_importance
import math

#####################################################################
def run_permutation_importance(clf,
                               X_train,
                               y_train,
                               feature_names: list,
                               n_repeats: int = 30,
                               random_state: int = 42,
                               output_path: str
                               ) -> pd.DataFrame:
    """
    Run permutation importance and save results to CSV.
    """
    r = permutation_importance(clf, X_train, y_train,
                               n_repeats=n_repeats,
                               random_state=random_state)

    df = pd.DataFrame({"feature": feature_names,
                       "importance": r.importances_mean,
                       "std": r.importances_std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    df["importance_minus_2std"] = df["importance"] - 2 * df["std"]

    df.to_csv(output_path, index=False)
    print(f"Permutation importance saved -> {output_path}")
    return df


def run_ice_plots(model,
                  X_train: pd.DataFrame,
                  feature_list: list,
                  output_dir: str,
                  n_cols: int = 2,
                  subsample: float = 0.3,
                  random_state: int = 42,
                  ) -> None:
    """
    Generate ICE + PDP mean plots
    """

    n_rows = math.ceil(len(feature_list) / n_cols)
    class_labels = {0: "DIA-", 1: "DIA+"}

    for target_class, label in class_labels.items():
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                                 figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        PartialDependenceDisplay.from_estimator(model,
                                                X_train,
                                                features=feature_list,
                                                target=target_class,
                                                kind="both",
                                                subsample=subsample,
                                                random_state=random_state,
                                                ax=axes[:len(feature_list)],
                                                ice_lines_kw={"color": "#85B7EB", "alpha": 0.3, "linewidth": 0.8},
                                                pd_line_kw={"color": "#185FA5", "linewidth": 2.0},
                                                )

        # hide any unused axes
        for ax in axes[len(feature_list):]:
            ax.set_visible(False)

        fig.suptitle(f"ICE + PDP — {label}", fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        out_path = os.path.join(output_dir, f"ice_class_{target_class}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved in {out_path}")