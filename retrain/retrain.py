import pandas as pd
import joblib
import mlflow

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


DATA_PATH = "data/raw/bitext_customer_support.csv"
MODEL_PATH = "artifacts/classifier.pkl"
ENCODER_PATH = "artifacts/encoder.pkl"


def retrain():

    df = pd.read_csv(DATA_PATH)

    X = df["instruction"].tolist()
    y = df["intent"].tolist()

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    X_emb = encoder.encode(X, show_progress_bar=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_emb, y, test_size=0.2, random_state=42
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")

    with mlflow.start_run():

        mlflow.log_metric("f1", f1)
        mlflow.sklearn.log_model(clf, "model")

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    print("Retraining done. F1:", f1)


if __name__ == "__main__":
    retrain()

