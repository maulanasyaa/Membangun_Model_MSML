import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

# models
models = [
    ("logistic_regression", LogisticRegression(solver="liblinear", random_state=42)),
    ("svm", SVC(kernel="rbf", C=1, gamma="scale", class_weight="balanced")),
    (
        "xgboost",
        XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
        ),
    ),
]

# datasets
df = pd.read_csv(
    "../Membangun_model/predict_the_introverts_from_the_extroverts_preprocessing/train_preprocessing.csv"
)

# separate features
X = df.drop(columns=["id", "Personality"], axis=1)
y = df["Personality"]


def modelling(X, y, models: list):
    # train test split
    print(f"Train test split..")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # setup mlflow
    mlflow.set_tracking_uri("../mlruns")
    mlflow.set_experiment("Predict_the_introvert_from_the_extrovert_modelling")

    print("Modelling..")
    for name, model in models:
        with mlflow.start_run(run_name=f"modelling_{name}"):
            mlflow.autolog()
            # modelling
            pipeline = Pipeline(steps=[(name, model)])
            pipeline.fit(X_train, y_train)

            # train score
            y_pred_train = pipeline.predict(X_train)
            score_train = accuracy_score(y_train, y_pred_train)
            mlflow.log_metric("manual_score_train", score_train)

            # test score
            y_pred_test = pipeline.predict(X_test)
            score_test = accuracy_score(y_test, y_pred_test)
            mlflow.log_metric("manual_score_test", score_test)

    print("Modelling complete.")


if __name__ == "__main__":
    modelling(X, y, models)
