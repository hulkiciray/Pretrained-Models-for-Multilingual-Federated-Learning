from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import logging
import warnings
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForMaskedLM
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    #filename="results.log",
    #filemode="w"
    handlers=[logging.StreamHandler()]  # Print to terminal
)

import os
os.environ["RAY_DEDUP_LOGS"] = "0"

#tokenizer=AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")


def gather_data(data_path: str = 'splits_data', verbose: bool = False) -> dict:
    train_files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('train.csv')])
    test_files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('test.csv')])
    dev_files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('dev.csv')])

    path_list = []
    for i in range(len(train_files)):
        train_path = train_files[i]
        dev_path = dev_files[i]
        test_path = test_files[i]
        path_set = (train_path, dev_path, test_path)
        path_list.append(path_set)

    all_files = {}
    for path in path_list:
        train_path, dev_path, test_path = path
        lang_name = train_path.split("/")[-1]
        lang_name = lang_name.split("_")[0]
        train_data = pd.read_csv(train_path, header=0, index_col=None)["text"].tolist()[:50]
        dev_data = pd.read_csv(dev_path, header=0, index_col=None)["text"].tolist()[:50]
        test_data = pd.read_csv(test_path, header=0, index_col=None)["text"].tolist()
        all_files[lang_name] = train_data, dev_data, test_data
        if verbose:
            logging.info(f"Reading {lang_name} with training length: {len(train_data)}, "
                         f"dev length: {len(dev_data)}, "
                         f"test length: {len(test_data)} completed")
    return all_files


class MLMDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.tokenizer(self.texts[idx], truncation=True, max_length=128, return_special_tokens_mask=True)


def MLMDataCreator(
        datasets_tuple: tuple,
        tokenizer,
        mlm_probability=0.15,
        train_batch_size=16,
        val_batch_size=16,
        test_batch_size=16
):
    train_texts, val_texts, test_texts = datasets_tuple

    # Create datasets
    train_dataset = MLMDataset(train_texts, tokenizer)
    val_dataset = MLMDataset(val_texts, tokenizer)
    test_dataset = MLMDataset(test_texts, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        collate_fn=data_collator,
        # huggingface datacollator aslında torch'taki collate_fn gibi düşünebiliriz. Batch'leri oluşturuken nasıl oluştururuzu verir.
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        collate_fn=data_collator
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        collate_fn=data_collator
    )

    return train_loader, val_loader, test_loader


def partition_dataloader(dataset_dict: dict, tokenizer, verbose: bool = False):
    # Create train-test-val dataloaders for each corpus
    lang_no_dict = {}
    dataset_loaders = {}
    keys = dataset_dict.keys()
    for idx, key in enumerate(keys):
        lang_no_dict[idx] = key
        #logging.info(f"dataset_dict[key]: {dataset_dict[key]}")
        train_loader, val_loader, test_loader = MLMDataCreator(datasets_tuple=dataset_dict[key], tokenizer=tokenizer)
        dataset_loaders[idx] = {"train_loader": train_loader, "val_loader": val_loader, "test_loader": test_loader}
    if verbose:
        logging.info(f"lang_no_dict: {lang_no_dict}")
    return dataset_loaders, lang_no_dict


#def load_partitions(dataset_loaders, partition_id: int, verbose: bool = False):
def load_partitions(partition_id: int, verbose: bool = False):
    dataset_dict = gather_data(verbose=verbose) # Default path is "splits_data"
    dataset_loaders, lang_no_dict = partition_dataloader(dataset_dict=dataset_dict, tokenizer=tokenizer, verbose=verbose)
    #logging.info(f"Language no: {partition_id} || Language name {lang_no_dict[partition_id]}")
    train_loader = dataset_loaders[partition_id]["train_loader"]
    val_loader = dataset_loaders[partition_id]["val_loader"]
    test_loader = dataset_loaders[partition_id]["test_loader"]
    if verbose:
        logging.info(f"train_loader, val_loader and test_loader fetched successfully!")
    return train_loader, val_loader, test_loader


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights_old(model: torch.nn.ModuleList, weights) -> None: # This is the older version
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.Tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)

def set_weights(model: torch.nn.ModuleList, weights) -> None: # This is better version. But it all depends on the model type. Either pytorch, tensorflow ...etc
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def train(
        model,
        train_dataloader,
        val_dataloader,
        epoch,
        device,
        optimizer,
        partition_id
):
    model.to(device)
    model.train()
    #model = DDP(model, device_ids=[device])
    model.zero_grad()
    losses = list()
    #logging.info("Model is in training mode for the client number {}".format(partition_id))
    for epoch in range(epoch):
        #logging.info(f"Training epoch number {epoch+1} starts for the client number {partition_id}")
        batch_losses = list()
        #batches = tqdm(train_dataloader, desc="Training for client {}".format(partition_id), colour="green")
        with tqdm(train_dataloader, desc="Training for client {}".format(partition_id), colour="green") as batches:
            for batch_idx, batch in enumerate(batches):
                # batch = {k: v.to(device) for k, v in batch.items()}
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                attn_mask = batch['attention_mask'].to(device)

                x = {"input_ids": input_ids,
                     "labels": labels,
                     "attention_mask": attn_mask}

                output = model(**x)
                loss = output.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_losses.append(output.loss.cpu().detach().repeat(len(batch)))
                losses.append(output.loss.cpu().detach().repeat(len(batch)))

                #if (batch_idx + 1) % 3 == 0:
                #    mean_batch_loss = torch.cat(batch_losses).mean()
                #    logging.info(
                #        f"client: {partition_id} ||epoch: {epoch+1} || batch: {batch_idx}/{len(train_dataloader)} || current_loss: {loss.item()} || mean_batch_loss: {mean_batch_loss.detach().item()}")

        logging.info(f"Client {partition_id} epoch {epoch+1} average epoch loss: {torch.cat(batch_losses).mean().item()}")
    mean_train_loss = torch.cat(losses).mean()
    val_loss, perplexity = test(model=model, test_dataloader=val_dataloader, device=device, partition_id=partition_id)
    logging.info(f"Client no: {partition_id} || Epoch no: {epoch+1} || Avg epoch loss: {mean_train_loss} || perplexity: {perplexity}")
    return val_loss, perplexity

def test(model, test_dataloader, device, partition_id):
    model.to(device)
    model.eval()
    test_losses = list()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            input_ids = batch['input_ids']
            labels = batch['labels']
            attn_mask = batch['attention_mask']

            x = {"input_ids": input_ids,
                 "labels": labels,
                 "attention_mask": attn_mask}
            output = model(**x)
            loss = output.loss
            test_losses.append(loss.repeat(len(batch)))
    mean_test_loss = torch.cat(test_losses).mean()
    logging.info(f"Mean test loss for the client {partition_id}: {mean_test_loss.item()}, perplexity: {torch.exp(mean_test_loss).detach().item()}")
    return mean_test_loss.item(), torch.exp(mean_test_loss).detach().item()