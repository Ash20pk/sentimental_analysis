"""Run model inference"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import argparse
import asyncio
import json
import math

import nada_numpy as na
import nada_numpy.client as na_client
import numpy as np
import py_nillion_client as nillion
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.crypto.keypairs import PrivateKey
from dotenv import load_dotenv
from nillion_python_helpers import create_nillion_client, create_payments_config
from py_nillion_client import NodeKey, UserKey

from common.utils import compute, store_secret_array

PARSER = argparse.ArgumentParser()
PARSER.add_argument(
    "--features-path",
    dest="features_path",
    type=str,
    required=True,
)
PARSER.add_argument(
    "--in-path",
    dest="in_path",
    type=str,
    required=True,
)
ARGS = PARSER.parse_args()

home = os.getenv("HOME")
load_dotenv(f"{home}/sentimental_analysis/nillion-testnet.env")

def interpret_sentiment(probability, positive_threshold=0.6, negative_threshold=0.4):
    if probability >= positive_threshold:
        return "positive"
    elif probability <= negative_threshold:
        return "negative"
    else:
        return "neutral"


async def main(features_path: str, in_path: str) -> None:
    """Main nada program"""

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

    # This information was provided by the model provider
    with open(in_path, "r") as provider_variables_file:
        provider_variables = json.load(provider_variables_file)

    program_id = provider_variables["program_id"]
    model_store_id = provider_variables["model_store_id"]
    model_provider_party_id = provider_variables["model_provider_party_id"]

    features = np.load(features_path)

    permissions = nillion.Permissions.default_for_user(model_user_client.user_id)
    permissions.add_compute_permissions({model_user_client.user_id: {program_id}})

    print("Storing input data...")

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

    print("Input data stored successfully!")

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

    logit = na_client.float_from_rational(result["logit_0"])
    print(f"Raw logit value: {logit}")

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    probability = sigmoid(logit)
    sentiment = interpret_sentiment(probability)
    
    print(f"🖥️  The sentiment probability is {probability:.4f} @ {na.get_log_scale()}-bit precision")
    print(f"🖥️  The predicted sentiment is {sentiment}")

    if sentiment == "neutral":
        print("The model is uncertain about this prediction.")
    elif sentiment == "positive":
        print(f"The model is {(probability - 0.6) / 0.4 * 100:.2f}% confident that the sentiment is positive.")
    else:
        print(f"The model is {(0.4 - probability) / 0.4 * 100:.2f}% confident that the sentiment is negative.")


if __name__ == "__main__":
    asyncio.run(main(ARGS.features_path, ARGS.in_path))
