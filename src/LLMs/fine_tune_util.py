import torch
from matplotlib import pyplot as plt
import numpy as np


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    '''pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels'''
    return logits # torch.argmax(logits, dim=-1)  

# this is causing massive memory usage because it accumulates all of the tensors before evaluating them (dumb!), there is a fix here that we have to implement
    #https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/12 - morenolq's answer
def compute_metrics(eval_preds):
    logits, labels = eval_preds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits)
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels)


    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Calculate loss
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # Calculate perplexity
    perplexity = torch.exp(loss)
    
    return {"eval_loss": loss.item(), "perplexity": perplexity.item()}

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