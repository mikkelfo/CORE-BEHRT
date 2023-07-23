from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, precision_score, recall_score

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
    