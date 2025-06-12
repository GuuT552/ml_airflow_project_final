import joblib
from sklearn.linear_model import LogisticRegression

def train_model(X, y, model_path='results/model.pkl'):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    joblib.dump(model, model_path)