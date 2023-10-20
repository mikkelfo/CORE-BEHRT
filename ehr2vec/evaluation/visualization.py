from common.config import instantiate
from sklearn.manifold import TSNE
from umap import UMAP


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