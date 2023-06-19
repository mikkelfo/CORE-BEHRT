from sklearn.manifold import TSNE


def dimensionality_reduction(embedding):
    tsne = TSNE(n_components=2, random_state=0)
    X = tsne.fit_transform(embedding)
    x, y = X[:, 0], X[:, 1]
    return x, y
