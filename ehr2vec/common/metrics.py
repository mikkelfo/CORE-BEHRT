from sklearn.metrics import (auc, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, confusion_matrix)


def pr_auc(model, X, y):
    y_scores = model.predict_proba(X)[:, 1]
    precision, recall, _ = precision_recall_curve(y, y_scores)
    return auc(recall, precision)

def roc_auc(model, X, y):
    y_pred_proba = model.predict_proba(X)[:,1]
    return roc_auc_score(y, y_pred_proba)

def precision(model, X, y):
    y_scores = model.predict(X)
    return precision_score(y, y_scores)

def recall(model, X, y):
    y_scores = model.predict(X)
    return recall_score(y, y_scores)

def f1(model, X, y):
    y_scores = model.predict(X)
    return f1_score(y, y_scores)

def top_k_precision(model, X, y, k=100):
    y_scores = model.predict_proba(X)[:, 1]
    top_k_indices = y_scores.argsort()[-k:][::-1]
    return precision_score(y[top_k_indices], model.predict(X[top_k_indices]))

def top_k_recall(model, X, y, k=100):
    y_scores = model.predict_proba(X)[:, 1]
    top_k_indices = y_scores.argsort()[-k:][::-1]
    return recall_score(y[top_k_indices], model.predict(X[top_k_indices]))

def top_k_f1(model, X, y, k=100):
    y_scores = model.predict_proba(X)[:, 1]
    top_k_indices = y_scores.argsort()[-k:][::-1]
    return f1_score(y[top_k_indices], model.predict(X[top_k_indices]))
    