import mlflow
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 37

    X_train = pd.read_csv("star_classification_preprocessing/X_train.csv")
    X_test = pd.read_csv("star_classification_preprocessing/X_test.csv")
    y_train = pd.read_csv("star_classification_preprocessing/y_train.csv")["class"]
    y_test = pd.read_csv("star_classification_preprocessing/y_test.csv")["class"]

    input_example = X_train.iloc[:5]

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

        model.fit(X_train, y_train)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
        )

        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
