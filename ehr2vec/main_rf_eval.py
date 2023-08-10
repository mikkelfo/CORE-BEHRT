from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score
from common.io import PatientHDF5Reader

reader = PatientHDF5Reader('outputs/pretraining/test/encodings/censored_patients2/encodings.h5')
# Train the random forest classifier with OOB scoring enabled
X, y = reader.read_arrays()

clf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
clf.fit(X, y)

# Use OOB predictions as a form of "validation"
oob_decision_function = clf.oob_decision_function_
oob_predictions = oob_decision_function.argmax(axis=1)
oob_probabilities = oob_decision_function[:, 1]

# Calculate metrics
roc_auc = roc_auc_score(y, oob_probabilities)
pr_auc = average_precision_score(y, oob_probabilities)
accuracy = accuracy_score(y, oob_predictions)
precision = precision_score(y, oob_predictions)
recall = recall_score(y, oob_predictions)

print(f'OOB ROC-AUC: {roc_auc}')
print(f'OOB PR-AUC: {pr_auc}')
print(f'OOB Accuracy: {accuracy}')
print(f'OOB Precision: {precision}')
print(f'OOB Recall: {recall}')