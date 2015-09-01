from sklearn_feature_prep import load_features
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score

ids, x, y = load_features()

model = LogisticRegression()
auc = cross_val_score(model, x, y, scoring='roc_auc', cv=5, n_jobs=4)

print(auc)