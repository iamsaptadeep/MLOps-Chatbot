import os
import joblib
import pandas as pd
import numpy as np
import logging
import random

from fastapi import FastAPI
from pydantic import BaseModel


# -------------------------------------------------
# Logging (stdout -> docker logs)
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")


# -------------------------------------------------
# App
# -------------------------------------------------
app = FastAPI(title="AI Support Chatbot")


# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(ROOT_DIR, "artifacts", "classifier.pkl")
ENCODER_PATH = os.path.join(ROOT_DIR, "artifacts", "encoder.pkl")
DATA_PATH = os.path.join(ROOT_DIR, "data", "raw", "bitext_customer_support.csv")


# -------------------------------------------------
# Config
# -------------------------------------------------
EXPECTED_EMB_DIM = 384
CONFIDENCE_THRESHOLD = 0.65


# Business rules from dataset patterns
KEYWORD_RULES = {
    "refund": "get_refund",
    "return": "get_refund",
    "money back": "get_refund",

    "cancel order": "cancel_order",
    "cancel": "cancel_order",

    "track": "track_order",
    "shipment": "track_order",
    "delivery": "track_order",

    "password": "reset_password",
    "forgot password": "reset_password",
    "login": "reset_password",
    "sign in": "reset_password",
}


# -------------------------------------------------
# Load Model
# -------------------------------------------------
try:
    classifier = joblib.load(MODEL_PATH)
    logger.info("Loaded classifier: %s", MODEL_PATH)

except Exception as e:

    logger.exception("Classifier load failed, using stub: %s", e)

    class _StubClassifier:

        def __init__(self):
            self.classes_ = np.array(["unknown"])

        def predict_proba(self, X):
            n = X.shape[0]
            return np.ones((n, 1))

        def predict(self, X):
            return np.array(["unknown"] * X.shape[0])

    classifier = _StubClassifier()


# -------------------------------------------------
# Load Encoder
# -------------------------------------------------
try:
    encoder = joblib.load(ENCODER_PATH)
    logger.info("Loaded encoder: %s", ENCODER_PATH)

except Exception as e:

    logger.exception("Encoder load failed, using stub: %s", e)

    class _StubEncoder:

        def encode(self, texts, **kwargs):
            n = len(texts)
            return np.zeros((n, EXPECTED_EMB_DIM))

    encoder = _StubEncoder()


# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
try:
    data = pd.read_csv(DATA_PATH)

    # Multiple responses per intent
    intent_map = (
        data
        .groupby("intent")["response"]
        .apply(list)
        .to_dict()
    )

    logger.info("Loaded %d intents", len(intent_map))

except Exception as e:

    logger.exception("Dataset load failed: %s", e)

    intent_map = {
        "unknown": ["Sorry, I couldn't understand."]
    }


# -------------------------------------------------
# Schemas
# -------------------------------------------------
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    intent: str
    response: str
    confidence: float


# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.get("/")
def health():
    return {"status": "running"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    try:

        text = req.message.lower().strip()


        # =============================================
        # 1️⃣ Rule-Based Override (Business Layer)
        # =============================================
        for key, forced_intent in KEYWORD_RULES.items():

            if key in text:

                logger.info("Rule override: '%s' -> %s", key, forced_intent)

                responses = intent_map.get(
                    forced_intent,
                    ["Please contact support for assistance."]
                )

                return {
                    "intent": forced_intent,
                    "response": random.choice(responses),
                    "confidence": 0.95
                }


        # =============================================
        # 2️⃣ Encode
        # =============================================
        emb = encoder.encode([req.message])
        emb = np.asarray(emb)

        if emb.ndim == 1:
            emb = emb.reshape(1, -1)


        # Fix dimension if needed
        if emb.shape[1] != EXPECTED_EMB_DIM:

            logger.warning(
                "Embedding dim mismatch: %d != %d",
                emb.shape[1],
                EXPECTED_EMB_DIM
            )

            if emb.shape[1] < EXPECTED_EMB_DIM:

                pad = EXPECTED_EMB_DIM - emb.shape[1]

                emb = np.pad(
                    emb,
                    ((0, 0), (0, pad)),
                    mode="constant"
                )

            else:

                emb = emb[:, :EXPECTED_EMB_DIM]


        # =============================================
        # 3️⃣ Predict
        # =============================================
        probs = classifier.predict_proba(emb)[0]

        idx = int(np.argmax(probs))

        intent = str(classifier.classes_[idx])

        confidence = float(probs[idx])


        # =============================================
        # 4️⃣ Confidence Gate
        # =============================================
        if confidence < CONFIDENCE_THRESHOLD:

            logger.info("Low confidence %.3f → escalate", confidence)

            return {
                "intent": "contact_human_agent",
                "response": "I'm not fully sure. Let me connect you with support.",
                "confidence": confidence
            }


        # =============================================
        # 5️⃣ Response Selection
        # =============================================
        responses = intent_map.get(
            intent,
            ["Sorry, I couldn't understand."]
        )

        response = random.choice(responses)


        return {
            "intent": intent,
            "response": response,
            "confidence": confidence
        }


    except Exception as e:

        logger.exception("Chat error: %s", e)

        return {
            "intent": "unknown",
            "response": "Internal error — please try again later.",
            "confidence": 0.0
        }

