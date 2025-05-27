from utils import (
    load_partitions,
    train,
    test,
    get_weights,
    set_weights
)

import torch
from torch.optim import AdamW
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    DistilBertForMaskedLM
)
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")
import logging
import json
from random import random
#from ray import RAY_verbose_spill_logs
#RAY_verbose_spill_logs=0
import os
os.environ["RAY_DEDUP_LOGS"] = "0"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    #filename="results.log",
    #filemode="w" # print to log file if you want
    handlers=[logging.StreamHandler()]  # Print to terminal. Can not be used with .log file
)

# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(self, model, device, optimizer, train_loader, val_loader, test_loader, partition_id, local_epochs):
        self.net = model
        self.local_epochs = local_epochs
        self.partition_id = partition_id
        self.device = device
        self.train_loader, self.val_loader, self.test_loader = train_loader, val_loader, test_loader
        self.optimizer = optimizer

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_weights(self.net, parameters)
        print(config)
        train_loss, train_perplexity = train( # model, train_dataloader, val_dataloader, epoch, device, optimizer
            model=self.net,
            train_dataloader=self.train_loader,
            val_dataloader=self.val_loader,
            epoch=self.local_epochs,
            device=self.device,
            optimizer=self.optimizer,
            partition_id=self.partition_id
        )
        complex_metric = {"a": [1, 3, 6], "b": [2, 4, 8], "deneme": random()}
        complex_metric_str = json.dumps(complex_metric)
        return get_weights(self.net), len(self.train_loader.dataset), {"loss": train_loss, "perplexity": train_perplexity, "language_id": self.partition_id, "complex_metric": complex_metric_str}

    def fit_tutorial(self, parameters, config): # Hemen yukarıdaki açıklamaya örnek olarak geliştirilmiştir.
        set_weights(self.net, parameters)
        print(config)
        train_loss, train_perplexity = train(
            model=self.net,
            train_dataloader=self.train_loader,
            val_dataloader=self.val_loader,
            epoch=self.local_epochs,
            device=self.device,
            optimizer=self.optimizer,
            partition_id=self.partition_id
        )

        complex_metric = {"a": [1,3,6], "b": [2,4,8], "deneme": random()}
        complex_metric_str = json.dumps(complex_metric)
        return get_weights(self.net), len(self.train_loader.dataset), {"loss": train_loss, "complex_metric": complex_metric_str, "perplexity": train_perplexity}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_weights(self.net, parameters)
        loss, perplexity = test(self.net, self.test_loader, self.device, self.partition_id)

        return float(loss), len(self.test_loader.dataset), {"loss": loss, "perplexity": perplexity, "language_id": self.partition_id}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBertForMaskedLM.from_pretrained("distilbert-base-multilingual-cased")
    optimizer = AdamW(model.parameters(), lr=2e-5)
    partition_id = context.node_config["partition-id"]
    train_loader, val_loader, test_loader = load_partitions(partition_id)#, verbose=True)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(model, device, optimizer, train_loader, val_loader, test_loader, partition_id, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(client_fn)