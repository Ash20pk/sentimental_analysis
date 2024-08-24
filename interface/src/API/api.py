import os
import sys
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import json
from fastapi.middleware.cors import CORSMiddleware
import logging


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
import nada_numpy as na
import nada_numpy.client as na_client
import py_nillion_client as nillion
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.crypto.keypairs import PrivateKey
from dotenv import load_dotenv
from nada_ai.client import SklearnClient
from nillion_python_helpers import create_nillion_client, create_payments_config
from py_nillion_client import NodeKey, UserKey
from utils import compute, store_secret_array, store_program, store_secrets
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Replace with your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store model, client, and vectorizer information
program_id = None
model_store_id = None
model_user_client = None
payments_client = None
payments_wallet = None
cluster_id = None
vectorizer = None
model_provider_party_id = None
model_user_party_id = None

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    probability: float
    confidence: float

def text_to_features(text: str) -> np.ndarray:
    [features] = vectorizer.transform([text]).toarray().tolist()
    return np.array(features).astype(float)

def interpret_sentiment(probability, positive_threshold=0.6, negative_threshold=0.4):
    if probability >= positive_threshold:
        return "positive"
    elif probability <= negative_threshold:
        return "negative"
    else:
        return "neutral"

@app.on_event("startup")
async def startup_event():
    global cluster_id, model_user_client, payments_client, payments_wallet, vectorizer, model_user_party_id
    
    # Load environment variables
    home = os.getenv("HOME")
    load_dotenv(f"{home}/.config/nillion/nillion-devnet.env")

    cluster_id = os.getenv("NILLION_CLUSTER_ID")
    grpc_endpoint = os.getenv("NILLION_NILCHAIN_GRPC")
    chain_id = os.getenv("NILLION_NILCHAIN_CHAIN_ID")
    seed = "my_seed"
    model_user_userkey = UserKey.from_seed((seed))
    model_user_nodekey = NodeKey.from_seed((seed))
    model_user_client = create_nillion_client(model_user_userkey, model_user_nodekey)
    model_user_party_id = model_user_client.party_id

    payments_config = create_payments_config(chain_id, grpc_endpoint)
    payments_client = LedgerClient(payments_config)
    payments_wallet = LocalWallet(
        PrivateKey(bytes.fromhex(os.getenv("NILLION_NILCHAIN_PRIVATE_KEY_0"))),
        prefix="nillion",
    )

    # Load the vectorizer
    vectorizer = joblib.load("../Model/vectorizer.joblib")

    print("vectorizer loaded successfully!")

@app.post("/analyze_sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    try:
        # Convert text to features
        features = text_to_features(request.text)
        logger.debug(f"Features generated: {features}")

        # Load provider variables
        with open('../Model/identifiers.json', "r") as provider_variables_file:
            provider_variables = json.load(provider_variables_file)

        program_id = provider_variables["program_id"]
        model_store_id = provider_variables["model_store_id"]
        model_provider_party_id = provider_variables["model_provider_party_id"]

        logger.debug(f"Program ID: {program_id}")
        logger.debug(f"Model Store ID: {model_store_id}")
        logger.debug(f"Model Provider Party ID: {model_provider_party_id}")

        permissions = nillion.Permissions.default_for_user(model_user_client.user_id)
        permissions.add_compute_permissions({model_user_client.user_id: {program_id}})

        features_store_id = await store_secret_array(
            model_user_client,
            payments_wallet,
            payments_client,
            cluster_id,
            features,
            "my_input",
            na.SecretRational,
            1,
            permissions,
        )
        logger.debug(f"Features Store ID: {features_store_id}")

        compute_bindings = nillion.ProgramBindings(program_id)
        compute_bindings.add_input_party("Provider", model_provider_party_id)
        compute_bindings.add_input_party("User", model_user_party_id)
        compute_bindings.add_output_party("User", model_user_party_id)

        result = await compute(
            model_user_client,
            payments_wallet,
            payments_client,
            program_id,
            cluster_id,
            compute_bindings,
            [model_store_id, features_store_id],
            nillion.NadaValues({}),
            verbose=True,
        )
        logger.debug(f"Compute result: {result}")

        logit = na_client.float_from_rational(result["logit_0"])
        logger.debug(f"Logit: {logit}")

        probability = 1 / (1 + np.exp(-logit))
        logger.debug(f"Probability: {probability}")

        sentiment = interpret_sentiment(probability)
        logger.debug(f"Interpreted sentiment: {sentiment}")

        if sentiment == "neutral":
            confidence = 0
        elif sentiment == "positive":
            confidence = (probability - 0.6) / 0.4 * 100
        else:
            confidence = (0.4 - probability) / 0.4 * 100
        logger.debug(f"Confidence: {confidence}")

        response = SentimentResponse(
            sentiment=sentiment,
            probability=float(probability),
            confidence=float(confidence)
        )
        logger.debug(f"Response: {response}")

        return response

    except Exception as e:
        logger.exception("An error occurred during sentiment analysis")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)