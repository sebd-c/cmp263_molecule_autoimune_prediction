# cmp263_molecule_autoimune_prediction

Code for running and evaluating the CMP263 autoimmune dataset models.

The repository has two main runnable workflows:

- Supervised model training/evaluation through `src.run`
- Unsupervised KMeans clustering through `src.models.clustering.get_clustering`

## Environment

Create an environment from the project file:

```bash
conda env create -f environment.yml
conda activate autoimune-ml
```

All commands below should be run from the repository root:

```bash
cd /home/user/<REPO_ROOT>
```

## Repository Guide

```text
src/
  dataset/                       Input datasets and processed dataset
  pre_processing/                Dataset cleaning/preparation scripts
  models/
    train.py                     Model builders, parameter grids, CV helpers
    data_split.py                Train/test split helper
    test.py                      Model loading/prediction helpers
    clustering/get_clustering.py KMeans clustering workflow
  plotters/                      Plotting helpers
  interpretability_tools/        Permutation importance, SHAP/ICE helpers
  eval/                          Evaluation metric helpers
  run.py                         Main supervised training pipeline

outputs/
  clustering/                    Clustering plots, labels, metrics, profiles
  figs/                          Existing generated figures
  models/                        Existing saved models
```

## Dataset

The main cleaned dataset is:

```text
src/dataset/fixed_dataset.csv
```

The supervised pipeline expects a target column named:

```text
Label
```

The clustering workflow drops `Label` before clustering if that column exists.

## Run The Supervised Pipeline

The supervised pipeline is defined in `src/run.py`.

Before running, check these constants near the top of `src/run.py`:

<INPUT_PATH> and <OUTPUT_DIR> should be altered inside .py scripts

Run:

```bash
python3 -m src.run
```

This pipeline:

- Loads `fixed_dataset.csv`
- Splits the data into train/test sets
- Trains/evaluates Decision Tree, Random Forest, KNN, SVC, and XGBoost
- Runs cross-validation
- Saves fitted models, selected features, metrics, and plots
- Runs permutation importance plots

Typical outputs include:

```text
outputs/*_cv_metrics.csv
outputs/*_model.joblib
outputs/*_selected_features.csv
outputs/*permutation_importance.csv
outputs/*permutation_importance.png
outputs/feature_summary_heatmap.png
```

## Run KMeans Clustering

The clustering workflow is defined in:

```text
src/models/clustering/get_clustering.py
```

Before running, edit these constants near the top of the file:


<INPUT_PATH> and <OUTPUT_DIR> should be altered inside .py scripts and other global variables include:
```python
LABEL_COLUMN = "Label"
N_CLUSTERS   = 2
K_MIN        = 2
K_MAX        = 10
```

Run:

```bash
python3 -m src.models.clustering.get_clustering
```

The clustering workflow:

- Loads the configured input CSV
- Drops `LABEL_COLUMN` if present
- Standardizes the features
- Generates elbow and silhouette plots for `K_MIN` through `K_MAX`
- Fits final KMeans using `N_CLUSTERS`
- Saves cluster labels, metrics, profile statistics, and a PCA scatter plot

Clustering outputs are saved to `OUTPUT_DIR`, currently:

```text
outputs/clustering/elbow.png
outputs/clustering/elbow.csv
outputs/clustering/silhouette_scores.png
outputs/clustering/silhouette_scores.csv
outputs/clustering/cluster_labels.csv
outputs/clustering/clustering_metrics.csv
outputs/clustering/cluster_profile.csv
outputs/clustering/kmeans_clusters_pca.png
outputs/clustering/kmeans_clusters_pca.csv
```

## Choosing The Number Of Clusters

Use the k-selection plots first:

```text
outputs/clustering/elbow.png
outputs/clustering/silhouette_scores.png
```

Then set cluster n in `src/models/clustering/get_clustering.py`, and rerun:

```bash
python3 -m src.models.clustering.get_clustering
```

