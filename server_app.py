import logging
import torch
import json
from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from utils import load_partitions
from transformers import AutoModelForMaskedLM, DistilBertForMaskedLM
from utils import get_weights, set_weights, test
from strategy import SaveModelStrategy

import os
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_verbose_spill_logs"] = "0"

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    language_ids = [m["language_id"] for num_examples, m in metrics]
    perplexities = [num_examples*metric["perplexity"] for num_examples, metric in metrics]

    for _, metric in metrics:
        logging.info("Language ID == {} || Perplexity == {} || Test Loss == {}".format(
            metric["language_id"],
            metric["perplexity"],
            metric["loss"])
        )
    return {"weighted_loss": sum(losses) / sum(examples), "weighted_perplexity": sum(perplexities) / sum(examples), "language_ids": language_ids}

def handle_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Tuple[int, List[Metrics]]:
    for _, metric in metrics:
        print(metric)

    return {}

def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds, one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}

def on_fit_config(num_rounds: int) -> Metrics:
    """With this call-back, we can adjust some default parameters of our training function"""
    if num_rounds > 2:
        return {"lr": num_rounds}

def get_evaluate_fn(testloader, device): # Bu fonksiyonun ismini biz belirledik
    """Return a call-back that evaluates the global model. This is being used for server-side evaluation.
    What I mean by that is we can evaluate our server model in every round with our custom dataset."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model using centralized dataset."""
        model = DistilBertForMaskedLM.from_pretrained("distilbert-base-multilingual-cased")
        set_weights(model, parameters_ndarrays)
        metric_dict = {}
        lang_dictionary = {0: 'cs', 1: 'de', 2: 'en', 3: 'es', 4: 'fi', 5: 'lt', 6: 'pl', 7: 'pt'}
        for key, value in lang_dictionary.items():
            train_loader, val_loader, test_loader = load_partitions(partition_id=key)
            loss, perplexity = test(model, train_loader, device=device, partition_id=key)
            metric_dict[value]=(server_round, loss, perplexity)
        loss = 1
        complex_metric_str = json.dumps(metric_dict)
        return loss, {"server-round, loss, perplexity": complex_metric_str}

    return evaluate

def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Initialize model parameters
    ndarrays = get_weights(AutoModelForMaskedLM.from_pretrained("distilbert-base-multilingual-cased"))
    parameters = ndarrays_to_parameters(ndarrays)

    # Load global test_loader for get_evaluate_fn function
    train_loader, val_loader, test_loader = load_partitions(partition_id=1)
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Define the strategy
    strategy = FedAvg(
        fraction_fit=context.run_config["fraction-fit"],
        fraction_evaluate=context.run_config["fraction-evaluate"],
        min_available_clients=context.run_config["min-available-clients"],
        initial_parameters=parameters,
        evaluate_fn=get_evaluate_fn(test_loader, device)
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)