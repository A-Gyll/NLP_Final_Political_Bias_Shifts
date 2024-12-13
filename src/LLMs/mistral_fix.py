# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset, DatasetDict
from fine_tune_util import compute_metrics, preprocess_logits_for_metrics
from datetime import datetime

# %%
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0).total_memory)

# %%
##################################### All Configuration ###################################
# This should be the only part of this code that is getting modified                      #
model_name = "meta-llama_Llama-3.2-3B" #"mistralai_Mistral-7B-v0_1"                                                  #
model_path = f"/home/puschb/UVA/NLP/NLP_Final_Political_Bias_Shifts/src/Local Models/{model_name}"                                              #                                                      
                                                                                          #
from_pretrained_params_dict = {                                                           #
    "pretrained_model_name_or_path" : model_path,                                         #
    #"load_in_8bit":True,                                                                 #
    "device_map":"auto",                                                                  #
    "torch_dtype": torch.float16                                                          #
}                                                                                         #
                                                                                          #
lora_config_params_dict = {                                                               #
    "lora_alpha":16,                                                                      #
    "lora_dropout":0.1,                                                                   #
    "r":64,                                                                               #
    "bias":"none",                                                                        #
    "task_type":TaskType.CAUSAL_LM,                                                       #
    #"target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"     #
     #               "down_proj", "lm_head"]                                               #
}                                                                                         #
                                                                                          #
quantization_params_dict = {                                                              #
                                                                                          #
}                                                                                         #
                                                                                          #
tokenizer_params_dict = {                                                                 #
  "truncation":False,
  "padding": False,
  "return_tensors": None,                                                                      #
  #"max_length":1024                                                                       #
}                                                                                         #
                                                                                          #
cur_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")                               #
checkpoint_dir = f"/home/puschb/UVA/NLP/NLP_Final_Political_Bias_Shifts/fine_tuned_llms/{model_name}/checkpoints/{cur_datetime}/"        #
metrics_dir = f"{checkpoint_dir}/metrics"                                                 #                                                                                          
                                                                                          #
training_args_dict = {                                                                    #                 
  "output_dir":checkpoint_dir,                                                            #                              
  "per_device_train_batch_size":8, # using A40 gpu, not sure if rivanna can handle more,  #                                                                                         
                                    # sticking with this for now                          #                                                                
  "per_device_eval_batch_size":8,                                                         #                                 
  #"num_train_epochs":3,                                                                  #                        
  "max_steps": 101,                                                                       #                   
  "evaluation_strategy":"steps",                                                          #                                
  "save_strategy":"steps",                                                                #                          
  "eval_steps":20,                                                                        #                  
  "save_steps":20,                                                                        #                  
  "load_best_model_at_end":True,                                                          #                                
  "metric_for_best_model":"perplexity",   # Select the best model based on perplexity     #                                                                                     
  "greater_is_better":False,              # Lower perplexity is better                    #                                                                      
  "logging_dir":metrics_dir,                                                              #                            
  "fp16":True,                                                                            #              
  #save_total_limit=3, # only keeping best 3                                              #                                            
}                                                                                         # 
###########################################################################################

# %%
#load dataset

seed = 210

data = pd.read_csv("/home/puschb/UVA/NLP/NLP_Final_Political_Bias_Shifts/data/Cleaned Data/CNN_comments_clean.csv")  
comments = data["comment"].astype(str).sample(frac=0.1, random_state=seed)

train_comments, test_comments = train_test_split(comments, test_size=0.3, random_state=seed)
val_comments, test_comments = train_test_split(test_comments, test_size=0.5, random_state=seed)

train_dataset = Dataset.from_pandas(pd.DataFrame({"text": train_comments}))
val_dataset = Dataset.from_pandas(pd.DataFrame({"text": val_comments}))
test_dataset = Dataset.from_pandas(pd.DataFrame({"text": test_comments}))

dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

# %%
# set up model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(**from_pretrained_params_dict)
peft_config =  LoraConfig(**lora_config_params_dict)
model = get_peft_model(model, peft_config)
print(model)

# %%
#set up tokenizer
tokenizer.pad_token = tokenizer.eos_token  # maybe this instead?: tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# using EOS should be fine since we want to talk like youtube comments
tokenizer.padding_side = "right"
def tokenize_function(examples):
    return tokenizer(examples["text"], **tokenizer_params_dict) # we don't do padding here, we let the data collater handle it

# %%
# Tokenize each split and remove the 'text' column
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Add 'labels' field for causal language modeling
#tokenized_datasets = tokenized_datasets.map(lambda examples: {"labels": examples["input_ids"]})
tokenized_datasets = tokenized_datasets.remove_columns(["__index_level_0__"])
print(tokenized_datasets)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

# %%
#print(tokenized_datasets['train']['labels'][0])

# %%
# set up trainer
training_args = TrainingArguments(**training_args_dict)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics   
)

# %%
trainer.train()


