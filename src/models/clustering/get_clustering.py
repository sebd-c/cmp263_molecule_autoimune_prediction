# -------------------------------------------------------------
# imports
# --------------------------------------------------------------
import os
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (calinski_harabasz_score,
                             davies_bouldin_score,
                             silhouette_samples,
                             silhouette_score,
                             )
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LINKAGE_METHOD  = "ward"
RANDOM_STATE    = 42

# Edit these values, then run:
# python3 -m src.models.clustering.get_clustering
INPUT_PATH      = "/home/debs/python_projects/cmp263---autoimune_dataset/src/dataset/fixed_dataset.csv"
OUTPUT_DIR      = "/home/debs/python_projects/cmp263---autoimune_dataset/outputs/clustering"
LABEL_COLUMN    = "Label"
N_CLUSTERS      = 2
K_MIN           = 2
K_MAX           = 10

# ---------------------------------------------------------------------------
# k-selection helpers
# ---------------------------------------------------------------------------

# elbow
def plot_elbow(X_scaled: np.ndarray,
               k_range: range = range(2, 11),
               output_path: str = "elbow.png",
               ) -> pd.DataFrame:
    """
    Fit KMeans for each k in k_range,
    record inertia,
    and plot the elbow curve.
    """
    records = []
    for k in k_range:
        km = KMeans(n_clusters=k,
                    random_state=RANDOM_STATE,
                    )
        km.fit(X_scaled)
        records.append({"k": k, "inertia": km.inertia_})

    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["k"], df["inertia"], marker="o",
            color="#185FA5", linewidth=1.8, markersize=6)
    ax.set_xlabel("Number of Clusters k", fontsize=11)
    ax.set_ylabel("Inertia within cluster", fontsize=11)
    ax.set_title("Elbow Curve", fontsize=12)
    ax.set_xticks(list(k_range))
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"elbow plot saved → {output_path}")
    return df

# silhouette
def plot_silhouette_scores(X_scaled: np.ndarray,
                           k_range: range = range(2, 11),
                           output_path: str = "silhouette_scores.png",
                           ) -> pd.DataFrame:
    """
    Compute mean silhouette score for each k and plot.
    """
    records = []
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE,
                        n_init="auto").fit_predict(X_scaled)
        score  = silhouette_score(X_scaled, labels)
        records.append({"k": k, "silhouette_mean": round(score, 4)})

    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["k"], df["silhouette_mean"], marker="o",
            color="#0F6E56", linewidth=1.8, markersize=6)
    ax.axhline(0, color="#E24B4A", lw=1, linestyle="--", alpha=0.6)
    ax.set_xlabel("Number of Clusters k", fontsize=11)
    ax.set_ylabel("Mean Silhouette Score", fontsize=11)
    ax.set_title("Silhouette Scores", fontsize=12)
    ax.set_xticks(list(k_range))
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"silhouette scores plot saved → {output_path}")
    return df

# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_kmeans(X_scaled: np.ndarray,
               n_clusters: int,
               random_state: int = RANDOM_STATE,
               ) -> KMeans:
    """
    Fit KMeans and return the fitted model (not just labels),
    so that inertia and cluster centers are accessible downstream.
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    km.fit(X_scaled)
    return km


def evaluate_clustering(X_scaled: np.ndarray,
                        labels: np.ndarray,
                        model_name: str = "kmeans",
                        ) -> dict:
    """
    Compute silhouette, Davies-Bouldin, and Calinski-Harabasz scores
    for KMeans labels.
    """
    n_clusters = len(set(labels))

    return {"model": model_name,
            "n_clusters": n_clusters,
            "silhouette": round(silhouette_score(X_scaled, labels), 4),
            "davies_bouldin": round(davies_bouldin_score(X_scaled, labels), 4),
            "calinski_harabasz": round(calinski_harabasz_score(X_scaled, labels), 4),
            }


def plot_kmeans_clusters(X_scaled: np.ndarray,
                         labels: np.ndarray,
                         output_path: str = "kmeans_clusters_pca.png",
                         random_state: int = RANDOM_STATE,
                         ) -> pd.DataFrame:
    """
    Project scaled features to 2 PCA dimensions and plot final KMeans labels.
    """
    pca = PCA(n_components=2, random_state=random_state)
    components = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "pc1": components[:, 0],
        "pc2": components[:, 1],
        "cluster": labels,
    })

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        pca_df["pc1"],
        pca_df["pc2"],
        c=pca_df["cluster"],
        cmap="tab10",
        s=36,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.4,
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=11)
    ax.set_title("KMeans Clusters (PCA Projection)", fontsize=12)
    legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    ax.add_artist(legend)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"kmeans cluster plot saved @ {output_path}")
    return pca_df


def build_cluster_profile(X_original: pd.DataFrame,
                          labels: np.ndarray,
                          model_name: str = "kmeans",
                          output_path: str = "cluster_profile.csv",
                          ) -> pd.DataFrame:
    """
    Compute per-cluster descriptive statistics on the original
    (unscaled) feature values and save to CSV.
    """
    df = X_original.copy()
    df["cluster"] = labels

    profile = (df.groupby("cluster").agg(["mean", "median", "std", "min", "max", "count"]))
    profile.columns = ["_".join(c) for c in profile.columns]
    profile = profile.reset_index()
    profile["cluster"] = profile["cluster"].astype(str)
    profile["model"] = model_name
    profile = profile[["model"] + [c for c in profile.columns if c != "model"]]

    profile.to_csv(output_path, index=False)
    print(f" cluster profile saved @ {output_path}")
    return profile


def run_clustering_pipeline(X: pd.DataFrame,
                            output_dir: str = "clustering",
                            k_range: range = range(2, 11),
                            n_clusters: int | None = None,
                            random_state: int = RANDOM_STATE,
                            ) -> dict:
    """
    Full KMeans clustering pipeline
    """
    os.makedirs(output_dir, exist_ok=True)
    X_scaled = StandardScaler().fit_transform(X)

    print("\n── k-selection plots ──")
    elbow_df = plot_elbow(X_scaled, k_range,
                          output_path=os.path.join(output_dir, "elbow.png"),
                          )
    elbow_df.to_csv(os.path.join(output_dir, "elbow.csv"), index=False)

    sil_df = plot_silhouette_scores(X_scaled, k_range,
                                    output_path=os.path.join(output_dir, "silhouette_scores.png"),
                                    )
    sil_df.to_csv(os.path.join(output_dir, "silhouette_scores.csv"), index=False)

    print("\n── fitting KMeans ──")
    km     = fit_kmeans(X_scaled, n_clusters, random_state)
    labels = km.labels_

    labels_df = X.copy()
    labels_df["cluster"] = labels
    labels_path = os.path.join(output_dir, "cluster_labels.csv")
    labels_df.to_csv(labels_path, index=False)
    print(f"cluster labels saved @ {labels_path}")

    plot_df = plot_kmeans_clusters(X_scaled, labels,
                                   output_path=os.path.join(output_dir, "kmeans_clusters_pca.png"),
                                   random_state=random_state,
                                   )
    plot_df.to_csv(os.path.join(output_dir, "kmeans_clusters_pca.csv"), index=False)

    print("\n── evaluation metrics ──")
    metrics = evaluate_clustering(X_scaled, labels)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, "clustering_metrics.csv"), index=False)
    print(f" metrics saved")
    print(metrics_df.to_string(index=False))

    print("\n── per-cluster profile ──")
    profile = build_cluster_profile(X_original=X,
        labels=labels,
        output_path=os.path.join(output_dir, "cluster_profile.csv"),
    )

    return {
        "model":   km,
        "labels":  labels,
        "plot":    plot_df,
        "metrics": metrics_df,
        "profile": profile,
    }


def main() -> None:
    """
    Run clustering from the editable constants at the top of this file.
    """
    data = pd.read_csv(INPUT_PATH)

    if LABEL_COLUMN in data.columns:
        X = data.drop(columns=[LABEL_COLUMN])
    else:
        X = data

    run_clustering_pipeline(
        X=X,
        output_dir=OUTPUT_DIR,
        k_range=range(K_MIN, K_MAX + 1),
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
    )


if __name__ == "__main__":
    main()
