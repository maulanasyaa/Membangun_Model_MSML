import dagshub
import mlflow
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

# setup dagshub
dagshub.init(repo_owner="maulanasyaa", repo_name="Membangun_Model_MSML", mlflow=True)

# datasets
df = pd.read_csv(
    "../Membangun_model/predict_the_introverts_from_the_extroverts_preprocessing/train_preprocessing.csv"
)

# separate features
X = df.drop(columns=["id", "Personality"], axis=1)
y = df["Personality"]


# function
def modelling(X, y):
    # train test split
    print("Train test split..")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # param distribution
    param_dist = {
        "n_estimators": randint(100, 1000),
        "learning_rate": uniform(0.01, 0.3),
        "max_depth": randint(3, 10),
        "min_child_weight": randint(1, 10),
        "gamma": uniform(0, 5),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
    }

    # scoring for random search
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted",
        "roc_auc": "roc_auc",
        "log_loss": "neg_log_loss",
    }

    # setup hyperparameter tuning
    random_search = RandomizedSearchCV(
        estimator=XGBClassifier(),
        param_distributions=param_dist,
        scoring=scoring,
        n_iter=20,
        refit="accuracy",
        cv=3,
        return_train_score=True,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )

    # setup mlflow
    mlflow.set_experiment("Predict_the_introvert_from_the_extrovert_modelling_tuning")

    print("Modelling..")
    with mlflow.start_run(run_name="modelling_xgboost_hyperparameter_tuning"):
        # hyperparameter tuning
        random_search.fit(X_train, y_train)

        # results
        results = random_search.cv_results_
        num_trials = len(results["params"])

        for i in range(num_trials):
            with mlflow.start_run(run_name=f"trial_{i + 1}", nested=True):
                # get params
                params = results["params"][i]
                mlflow.log_params(params)

                # metrics
                metrics_to_log = {
                    # autolog metrics
                    "training_accuracy_score": results["mean_train_accuracy"][i],
                    "training_precision_score": results["mean_train_precision"][i],
                    "training_recall_score": results["mean_train_recall"][i],
                    "training_f1_score": results["mean_train_f1"][i],
                    "training_roc_auc": results["mean_train_roc_auc"][i],
                    "training_log_loss": -results["mean_train_log_loss"][i],
                }

                mlflow.log_metrics(metrics_to_log)

                if i == random_search.best_index_:
                    mlflow.set_tag("candidate", "best")

        # get best model
        best_model = random_search.best_estimator_

        y_pred_test = best_model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred_test)

        mlflow.log_metric("test_accuracy_score", test_acc)

    print("Modelling complete.")


if __name__ == "__main__":
    modelling(X, y)
