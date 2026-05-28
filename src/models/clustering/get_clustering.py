# -------------------------------------------------------------
# imports
# --------------------------------------------------------------
import os
import warnings

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


def plot_dendrogram(X_scaled: np.ndarray,
                    method: str = LINKAGE_METHOD,
                    truncate_p: int = 30,
                    output_path: str = "dendrogram.png",
                    ) -> None:
    """
    Compute a linkage matrix and plot the hierarchical clustering dendrogram.
    Use the longest vertical gap (highest fusion distance jump) to pick k.
    """
    Z = linkage(X_scaled, method=method)

    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(
        Z,
        truncate_mode="lastp",
        p=truncate_p,
        leaf_rotation=90,
        leaf_font_size=9,
        show_contracted=True,
        ax=ax,
        color_threshold=0.7 * max(Z[:, 2]),
    )
    ax.set_xlabel("sample index (or cluster size)", fontsize=11)
    ax.set_ylabel(f"fusion distance ({method} linkage)", fontsize=11)
    ax.set_title(f"dendrogram — {method} linkage", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ dendrogram saved → {output_path}")


def suggest_k_from_dendrogram(X_scaled: np.ndarray,
                               method: str = LINKAGE_METHOD) -> int:
    """
    Heuristic: find the k suggested by the largest gap between consecutive
    fusion distances in the linkage matrix.

    Returns
    -------
    int
        Suggested number of clusters.
    """
    Z       = linkage(X_scaled, method=method)
    dists   = Z[:, 2]
    gaps    = np.diff(dists)
    best_k  = len(dists) - np.argmax(gaps[::-1]) + 1
    print(f"  ℹ dendrogram largest-gap heuristic suggests k = {best_k}")
    return best_k


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_kmeans(
    X_scaled: np.ndarray,
    n_clusters: int,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Fit KMeans and return cluster labels."""
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    return km.fit_predict(X_scaled)


def fit_agglomerative(
    X_scaled: np.ndarray,
    n_clusters: int,
    linkage_method: str = LINKAGE_METHOD,
) -> np.ndarray:
    """Fit AgglomerativeClustering and return cluster labels."""
    agg = AgglomerativeClustering(n_clusters=n_clusters,
                                  linkage=linkage_method)
    return agg.fit_predict(X_scaled)


def fit_dbscan(
    X_scaled: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
) -> np.ndarray:
    """
    Fit DBSCAN and return cluster labels.
    Label -1 indicates noise points (not assigned to any cluster).

    Parameters
    ----------
    eps : float
        Maximum distance between two samples to be considered neighbours.
    min_samples : int
        Minimum samples in a neighbourhood to form a core point.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"  ℹ DBSCAN found {n_clusters} clusters, {n_noise} noise points")
    return labels


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def evaluate_clustering(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    model_name: str,
) -> dict:
    """
    Compute silhouette, Davies-Bouldin, and Calinski-Harabasz scores.
    Skips gracefully if fewer than 2 clusters are found (e.g. DBSCAN edge case).

    Parameters
    ----------
    X_scaled : np.ndarray
    labels : np.ndarray
        Cluster label per sample. -1 (noise) points are excluded.
    model_name : str

    Returns
    -------
    dict
        Keys: model, n_clusters, n_noise, silhouette, davies_bouldin,
              calinski_harabasz.
    """
    mask       = labels != -1          # exclude DBSCAN noise
    X_eval     = X_scaled[mask]
    labels_eval = labels[mask]

    n_clusters = len(set(labels_eval))
    n_noise    = (labels == -1).sum()

    if n_clusters < 2:
        warnings.warn(f"{model_name}: fewer than 2 clusters — metrics not computed.")
        return {"model": model_name, "n_clusters": n_clusters,
                "n_noise": int(n_noise),
                "silhouette": None, "davies_bouldin": None,
                "calinski_harabasz": None}

    return {
        "model":              model_name,
        "n_clusters":         n_clusters,
        "n_noise":            int(n_noise),
        "silhouette":         round(silhouette_score(X_eval, labels_eval), 4),
        "davies_bouldin":     round(davies_bouldin_score(X_eval, labels_eval), 4),
        "calinski_harabasz":  round(calinski_harabasz_score(X_eval, labels_eval), 4),
    }


# ---------------------------------------------------------------------------
# Per-cluster profile
# ---------------------------------------------------------------------------

def build_cluster_profile(
    X_original: pd.DataFrame,
    labels: np.ndarray,
    model_name: str,
    output_path: str = "cluster_profile.csv",
) -> pd.DataFrame:
    """
    Compute per-cluster descriptive statistics on the original
    (unscaled) feature values.

    Noise points (label == -1) are included as cluster "-1 (noise)".

    Parameters
    ----------
    X_original : pd.DataFrame
        Unscaled feature matrix — use original units for interpretability.
    labels : np.ndarray
    model_name : str
    output_path : str

    Returns
    -------
    pd.DataFrame
        Multi-index DataFrame: (cluster, statistic) × feature.
    """
    df = X_original.copy()
    df["cluster"] = labels

    profile = (
        df.groupby("cluster")
          .agg(["mean", "median", "std", "min", "max", "count"])
    )
    profile.columns = ["_".join(c) for c in profile.columns]
    profile.insert(0, "model", model_name)
    profile.reset_index(inplace=True)
    profile["cluster"] = profile["cluster"].apply(
        lambda c: f"{c} (noise)" if c == -1 else str(c)
    )

    profile.to_csv(output_path, index=False)
    print(f"  ✓ cluster profile saved → {output_path}")
    return profile


# ---------------------------------------------------------------------------
# 2-D projection plot
# ---------------------------------------------------------------------------

def plot_clusters_pca(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    output_path: str = "clusters_pca.png",
) -> None:
    """
    Project X_scaled to 2 PCA components and scatter-plot cluster assignments.
    Noise points (DBSCAN label -1) are shown in gray.

    Parameters
    ----------
    X_scaled : np.ndarray
    labels : np.ndarray
    model_name : str
    output_path : str
    """
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_scaled)

    palette = [
        "#185FA5", "#0F6E56", "#993C1D", "#993556",
        "#534AB7", "#3B6D11", "#854F0B", "#A32D2D",
    ]

    unique = sorted(set(labels))
    fig, ax = plt.subplots(figsize=(7, 5))

    for i, cluster in enumerate(unique):
        mask  = labels == cluster
        color = "#888780" if cluster == -1 else palette[i % len(palette)]
        label = "noise" if cluster == -1 else f"cluster {cluster}"
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, label=label,
                   s=18, alpha=0.7, linewidths=0)

    ax.set_xlabel("PC 1", fontsize=11)
    ax.set_ylabel("PC 2", fontsize=11)
    ax.set_title(f"cluster projection (PCA) — {model_name}", fontsize=12)
    ax.legend(fontsize=9, markerscale=1.5)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ PCA cluster plot saved → {output_path}")


# ---------------------------------------------------------------------------
# Master function — run all three models
# ---------------------------------------------------------------------------

def run_clustering_pipeline(
    X: pd.DataFrame,
    output_dir: str = "clustering",
    k_range: range = range(2, 11),
    n_clusters: int | None = None,
    linkage_method: str = LINKAGE_METHOD,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Full clustering pipeline:
      1. Scale features
      2. Plot elbow, silhouette scores, dendrogram (k-selection)
      3. Fit KMeans, Agglomerative, DBSCAN
      4. Evaluate each model
      5. Save labels, metrics, per-cluster profile, and PCA plots

    Parameters
    ----------
    X : pd.DataFrame
        Unscaled feature matrix (only numeric columns).
    output_dir : str
        All outputs written here.
    k_range : range
        k values to evaluate for elbow and silhouette plots.
    n_clusters : int or None
        Number of clusters for KMeans and Agglomerative.
        If None, the dendrogram largest-gap heuristic is used.
    linkage_method : str
        Linkage for both dendrogram and AgglomerativeClustering.
    dbscan_eps : float
    dbscan_min_samples : int
    random_state : int

    Returns
    -------
    dict
        Keys: "labels" (dict of model → np.ndarray),
              "metrics" (pd.DataFrame),
              "profiles" (dict of model → pd.DataFrame).
    """
    os.makedirs(output_dir, exist_ok=True)
    X_scaled = StandardScaler().fit_transform(X)

    # ── Step 1 — k selection ──────────────────────────────────────────────────
    print("\n── k-selection plots ──")
    elbow_df = plot_elbow(
        X_scaled, k_range,
        output_path=os.path.join(output_dir, "elbow.png"),
    )
    elbow_df.to_csv(os.path.join(output_dir, "elbow.csv"), index=False)

    sil_df = plot_silhouette_scores(
        X_scaled, k_range,
        output_path=os.path.join(output_dir, "silhouette_scores.png"),
    )
    sil_df.to_csv(os.path.join(output_dir, "silhouette_scores.csv"), index=False)

    plot_dendrogram(
        X_scaled,
        method=linkage_method,
        output_path=os.path.join(output_dir, "dendrogram.png"),
    )

    # resolve k
    if n_clusters is None:
        n_clusters = suggest_k_from_dendrogram(X_scaled, method=linkage_method)
    print(f"  ℹ using k = {n_clusters} for KMeans and Agglomerative")

    # ── Step 2 — fit models ───────────────────────────────────────────────────
    print("\n── fitting models ──")
    labels_km  = fit_kmeans(X_scaled, n_clusters, random_state)
    labels_agg = fit_agglomerative(X_scaled, n_clusters, linkage_method)
    labels_db  = fit_dbscan(X_scaled, dbscan_eps, dbscan_min_samples)

    all_labels = {
        "kmeans":       labels_km,
        "agglomerative": labels_agg,
        "dbscan":       labels_db,
    }

    # ── Step 3 — save labels ──────────────────────────────────────────────────
    labels_df = X.copy()
    for name, lbl in all_labels.items():
        labels_df[f"cluster_{name}"] = lbl
    labels_path = os.path.join(output_dir, "cluster_labels.csv")
    labels_df.to_csv(labels_path, index=False)
    print(f"\n  ✓ cluster labels saved → {labels_path}")

    # ── Step 4 — evaluate ─────────────────────────────────────────────────────
    print("\n── evaluation metrics ──")
    metrics_rows = [
        evaluate_clustering(X_scaled, lbl, name)
        for name, lbl in all_labels.items()
    ]
    metrics_df   = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(output_dir, "clustering_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  ✓ metrics saved → {metrics_path}")
    print(metrics_df.to_string(index=False))

    # ── Step 5 — per-cluster profiles ─────────────────────────────────────────
    print("\n── per-cluster profiles ──")
    profiles = {}
    for name, lbl in all_labels.items():
        profiles[name] = build_cluster_profile(
            X_original=X,
            labels=lbl,
            model_name=name,
            output_path=os.path.join(output_dir, f"profile_{name}.csv"),
        )

    # ── Step 6 — PCA projection plots ─────────────────────────────────────────
    print("\n── PCA projection plots ──")
    for name, lbl in all_labels.items():
        plot_clusters_pca(
            X_scaled, lbl, name,
            output_path=os.path.join(output_dir, f"pca_{name}.png"),
        )

    return {
        "labels":   all_labels,
        "metrics":  metrics_df,
        "profiles": profiles,
    }
