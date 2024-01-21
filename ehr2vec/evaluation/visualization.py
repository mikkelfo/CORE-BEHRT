import matplotlib.colors as mcolors
import numpy as np
from sklearn.manifold import TSNE

from ehr2vec.common.config import instantiate

# from umap import UMAP


def dimensionality_reduction(embedding):
    tsne = TSNE(n_components=2, random_state=0)
    X = tsne.fit_transform(embedding)
    x, y = X[:, 0], X[:, 1]
    return x, y

def project_embeddings(data: dict, cfg)->dict:
    """Reduce dimensionality of concept_enc using methods to dims"""
    for _, value in cfg.project_methods.items():
        method = instantiate(value)
        for n in value.dims:
            proj = method.fit_transform(data['concept_enc'])
            for i in range(n):
                method_name = f'P_{str(method.__name__)}_{n}D_{i}'
                data[method_name] = proj[:,i]
    return data

def define_custom_sequential_colormap(color_start:str = "#FFFFFF", color_end:str="#20a4f3")->mcolors.ListedColormap:
    # Convert the hex colors to RGB values
    rgb_start = mcolors.hex2color(color_start)
    rgb_end = mcolors.hex2color(color_end)
    
    # Define the colors for the colormap
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(rgb_start[0], rgb_end[0], N)
    vals[:, 1] = np.linspace(rgb_start[1], rgb_end[1], N)
    vals[:, 2] = np.linspace(rgb_start[2], rgb_end[2], N)
    cmap = mcolors.ListedColormap(vals)
    return cmap
