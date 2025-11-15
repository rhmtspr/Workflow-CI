import mlflow
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Star Classification")

n_estimators = int(sys.argv[1])
max_depth = int(sys.argv[2])

X_train = pd.read_csv("star_classification_preprocessing/X_train.csv")
X_test = pd.read_csv("star_classification_preprocessing/X_test.csv")
y_train = pd.read_csv("star_classification_preprocessing/y_train.csv")["class"]
y_test = pd.read_csv("star_classification_preprocessing/y_test.csv")["class"]

input_example = X_train.iloc[:5]

with mlflow.start_run():
    mlflow.autolog()

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example,
    )

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
