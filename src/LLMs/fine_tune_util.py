import torch
from matplotlib import pyplot as plt
import numpy as np
import os
import random
from transformers import Trainer
import pandas as pd
import json
import math

def preprocess_logits_for_metrics(logits, labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = logits.to(device).float()
    labels = labels.to(device).long()

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    return loss

def preprocess_logits_for_metrics_mlm(logits, labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = logits.to(device).float()
    labels = labels.to(device).long()

    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

    return loss

# this is causing massive memory usage because it accumulates all of the tensors before evaluating them (dumb!), there is a fix here that we have to implement
    #https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/12 - morenolq's answer
def compute_metrics(eval_preds):
    losses = eval_preds[0]  
    if isinstance(losses, np.ndarray):
        losses = torch.tensor(losses, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    total_loss = torch.sum(losses)  
    perplexity = math.exp(total_loss.item() / len(losses))  

    return {
        "eval_loss": total_loss.item()/ len(losses),
        "perplexity": perplexity,
    }

def token_length_histogram(dataset, split_name):
    # Initialize a list to store token lengths
    token_lengths = []

    # Collect the lengths of all tokenized samples
    for example in dataset[split_name]:
        length = len(example["input_ids"])
        token_lengths.append(length)

    # Create a histogram of token lengths with bins of size 500
    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths, bins=range(0, max(token_lengths) + 100, 100), edgecolor='black')

    # Customize the plot
    plt.title(f"Token Length Distribution in {split_name} Split")
    plt.xlabel("Token Length (binned in chunks of 100)")
    plt.ylabel("Number of Samples")
    plt.grid(True)
    
    # Show the plot
    plt.show()

def make_serializable(obj):
    if isinstance(obj, (int, float, str, bool)):
        return obj
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (np.ndarray, torch.Tensor)):
        return obj.tolist()
    else:
        return str(obj)

def save_dicts_to_csv(dicts_dict, csv_path, model_name, experiment_id):
    dict_serializable = {key: make_serializable(value) for key, value in dicts_dict.items()}
   
    columns_order = ['model_name', 'experiment_id'] + list(dict_serializable.keys())
    dict_serializable['model_name'] = model_name
    dict_serializable['experiment_id'] = experiment_id

    new_row_df = pd.DataFrame([dict_serializable], columns=columns_order)

    # Set experiment_id as the index
    new_row_df.set_index('experiment_id', inplace=True)

    # Check if the CSV exists
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        existing_df.set_index('experiment_id', inplace=True)  # Set the same index for the existing CSV
        updated_df = pd.concat([existing_df, new_row_df], axis=0, ignore_index=False)
    else:
        updated_df = new_row_df

    updated_df.to_csv(csv_path)


def save_metrics(data, save_path):
    save_data = {}

    for entry in data:
        step = str(entry["step"])  
        for key, value in entry.items():
            if key == "step":
                continue  
            if key not in save_data:
                save_data[key] = {}
            save_data[key][step] = value

    with open(save_path, "w") as json_file:
        json.dump(save_data, json_file, indent=4)



# https://discuss.huggingface.co/t/evaluate-subset-of-data-during-training/10952/6
class EvalSampleDatasetTrainer(Trainer):
    def __init__(self, *args, eval_sample_size_proportion=.1, seed = 42, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_sample_size_proportion = eval_sample_size_proportion
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)

    def get_eval_dataloader(self, eval_dataset=None):
        '''
        Samples the evaluation dataset and returns a subset 
        of size self.eval_sample_size.
        '''
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        total_size = len(eval_dataset)
        sample_size = int(total_size * self.eval_sample_size_proportion)
        idxs = random.sample(range(total_size), sample_size)
        eval_subset = eval_dataset.select(idxs)
        return super().get_eval_dataloader(eval_subset)