# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import seaborn as sns
import plotly.express as px
from umap import UMAP

###########################################################################################

# PCA

# defining function
def run_pca(data: pd.DataFrame,
            feature_cols: list,
            n_components: int,
            random_state: int
            ) -> np.ndarray:
    # getting data from df
    features = data[feature_cols]

    # scaling data (converting values to z-scores)
    features = sklearn.preprocessing.StandardScaler().fit_transform(features)

    # getting PCA based on given components num
    pca = sklearn.decomposition.PCA(n_components=n_components,
                                    random_state=random_state)

    # getting principal components
    principal_components = pca.fit_transform(features)

    sil_score = sklearn.metrics.silhouette_score(principal_components, data['Label'])
    print(f'silhouette score is: {sil_score}')

    # getting explained variation per component
    var_per_components = pca.explained_variance_ratio_
    x_var, y_var, z_var = var_per_components
    x_var_percent = round(x_var * 100, 3)
    y_var_percent = round(y_var * 100, 3)
    z_var_percent = round(y_var * 100, 3)

    # getting x, y values
    x = principal_components[:, 0]
    y = principal_components[:, 1]
    z = principal_components[:, 2]

    # adding cols to df
    data['x'] = x
    data['y'] = y
    data['z'] = z

    # expected classifications
    # defining update dict
    label_dict = {0: 'DIA-',
                  1: 'DIA+'
                  }

    # updating labels col
    data.replace({'Label': label_dict},
                 inplace=True)

    # creating plot pc1 x pc2
    sns.scatterplot(data=data,
                x='x',
                y='y',
                hue='Label')

    # defining plot axis labels
    x_label = f'PC1 ({x_var_percent}%)'
    y_label = f'PC2 ({y_var_percent}%)'

    # updating plot axis labels
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # showing plot
    plt.show()

    plt.savefig('pca.svg')
    plt.savefig('pca.pdf')
  #   # creating plot pc1 x pc3
  #   scatterplot(data=data,
  #               x='x',
  #               y='z',
  #               hue='label')

  #   # defining plot axis labels
  #   x_label = f'PC1 ({x_var_percent}%)'
  #   y_label = f'PC3 ({z_var_percent}%)'

  #   # updating plot axis labels
  #   plt.xlabel(x_label)
  #   plt.ylabel(y_label)

  #   # showing plot
  #   plt.show()

  # # creating plot pc2 x pc3
  #   scatterplot(data=data,
  #               x='y',
  #               y='z',
  #               hue='label')

  #   # defining plot axis labels
  #   x_label = f'PC2 ({y_var_percent}%)'
  #   y_label = f'PC3 ({z_var_percent}%)'

  #   # updating plot axis labels
  #   plt.xlabel(x_label)
  #   plt.ylabel(y_label)

  #   # showing plot
  #   plt.show()


#########################################################################################

# TSNE

# defining function
def run_tsne(data: pd.DataFrame,
             feature_cols: list,
             n_components: int,
             perplexity: int,
             max_iter: int,
             random_state: int
             ) -> np.ndarray:
    # getting data from df
    features = data[feature_cols]

    # scaling data (converting values to z-scores)
    features = sklearn.preprocessing.StandardScaler().fit_transform(features)

    # getting TSNE based on given components num
    tsne = sklearn.manifold.TSNE(n_components=n_components,
                                 perplexity=perplexity,
                                 max_iter=max_iter,
                                 random_state=random_state)

    # getting principal components
    principal_components = tsne.fit_transform(features)

    kl_val = tsne.kl_divergence_
    print(f"KL div: {kl_val:.2f}")

    # Step 3: Apply clustering on t-SNE-transformed data
    kmeans = sklearn.cluster.KMeans(n_clusters=2, random_state=42)
    y_pred = kmeans.fit_predict(principal_components)

    # Step 4: Calculate silhouette metrics
    global_sil_score = sklearn.metrics.silhouette_score(principal_components, y_pred)
    print(f"Global Silhouette Score: {global_sil_score:.2f}")

    # getting x, y values
    x = principal_components[:, 0]
    y = principal_components[:, 1]

    # adding cols to df
    data['x'] = x
    data['y'] = y

    # expected classifications
    # defining update dict
    label_dict = {0: 'DIA-',
                  1: 'DIA+'
                  }

    # updating labels col
    data.replace({'Label': label_dict},
                 inplace=True)

    # creating plot
    ax = sns.scatterplot(data=data,
                         x='x',
                         y='y',
                         hue='Label')

    # defining plot axis labels
    x_label = f'T-SNE 1'
    y_label = f'T-SNE 2'

    # updating plot axis labels
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # showing plot
    plt.show()

    plt.savefig('tsne.svg')
    plt.savefig('tsne.pdf')

#######################################################################################
# UMAP

# defining function
def run_umap(data: pd.DataFrame,
             feature_cols: list,
             random_state: int
             ) -> np.ndarray:
    # getting data from df
    features = data[feature_cols]

    # scaling data (converting values to z-scores)
    features = sklearn.preprocessing.StandardScaler().fit_transform(features)

    # getting UMAP based on given components num
    data_umap = UMAP(random_state=random_state)

    # getting principal components
    principal_components = data_umap.fit_transform(features)

    # getting x, y values
    x = principal_components[:, 0]
    y = principal_components[:, 1]

    # adding cols to df
    data['x'] = x
    data['y'] = y

    # defining update dict
    label_dict = {0: 'DIA-',
                  1: 'DIA+'
                  }


    # updating labels col
    data.replace({'Label': label_dict},
                 inplace=True)

    # creating plot
    sns.scatterplot(data=data,
                    x='x',
                    y='y',
                    hue='Label')

    # defining plot axis labels
    x_label = f'UMAP 1'
    y_label = f'UMAP 2'

    # updating plot axis labels
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # showing plot
    plt.show()

    plt.savefig('umap.svg')
    plt.savefig('umap.pdf')