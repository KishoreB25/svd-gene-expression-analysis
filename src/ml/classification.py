# src/ml/classification.py

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def train_logistic_regression(X_train, y_train):
    """
    Trains multinomial logistic regression.
    """
    model = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs"
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates classification performance.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return acc, cm
