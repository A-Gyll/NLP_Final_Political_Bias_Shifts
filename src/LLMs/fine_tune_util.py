import torch
def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels

# this is causing massive memory usage because it accumulates all of the tensors before evaluating them (dumb!), there is a fix here that we have to implement
    #https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/12 - morenolq's answer
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Calculate loss
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # Calculate perplexity
    perplexity = torch.exp(loss)
    
    return {"eval_loss": loss.item(), "perplexity": perplexity.item()}