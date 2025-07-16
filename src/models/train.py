from sklearn.linear_model import LogisticRegression


def train_logistic_regression(X_train, y_train):
    """
    Trains a logistic regression model using the provided training data.

    Parameters:
        X_train (array-like): Training feature data.
        y_train (array-like): Training target labels.

    Returns:
        model (LogisticRegression): Trained logistic regression model.
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def test_logistic_regression(model, X_test, y_test):
    """
    Evaluates a trained logistic regression model on the provided test data.

    Parameters:
        model (LogisticRegression): Trained logistic regression model.
        X_test (array-like): Test feature data.
        y_test (array-like): Test target labels.

    Returns:
        accuracy (float): Accuracy of the model on the test data.
        predictions (array-like): Predicted labels for the test data.
    """
    accuracy = model.score(X_test, y_test)
    predictions = model.predict(X_test)
    return accuracy, predictions


