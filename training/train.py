import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from sentence_transformers import SentenceTransformer


DATA_PATH = "data/raw/bitext_customer_support.csv"
MODEL_DIR = "artifacts"

TEXT_COL = "instruction"   # may change after checking CSV
LABEL_COL = "intent"


def main():

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("Columns:", df.columns)

    # Keep only needed columns
    df = df[[TEXT_COL, LABEL_COL]].dropna()

    X = df[TEXT_COL].values
    y = df[LABEL_COL].values

    print("Total samples:", len(df))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Loading sentence transformer...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding text...")
    X_train_emb = encoder.encode(X_train, show_progress_bar=True)
    X_test_emb = encoder.encode(X_test, show_progress_bar=True)

    print("Training classifier...")
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train_emb, y_train)

    print("Evaluating...")
    preds = clf.predict(X_test_emb)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    print("Accuracy:", acc)
    print("Macro F1:", f1)

    # MLflow tracking
    mlflow.set_experiment("chatbot-intent-classifier")

    with mlflow.start_run():

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("encoder", "all-MiniLM-L6-v2")
        mlflow.log_param("test_size", 0.2)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("macro_f1", f1)

        # Save artifacts
        import os
        os.makedirs(MODEL_DIR, exist_ok=True)

        joblib.dump(clf, f"{MODEL_DIR}/classifier.pkl")
        joblib.dump(encoder, f"{MODEL_DIR}/encoder.pkl")

        mlflow.log_artifact(f"{MODEL_DIR}/classifier.pkl")
        mlflow.log_artifact(f"{MODEL_DIR}/encoder.pkl")

    print("Training complete. Model saved.")


if __name__ == "__main__":
    main()
