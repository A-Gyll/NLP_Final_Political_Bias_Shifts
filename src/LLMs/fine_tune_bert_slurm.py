import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset, DatasetDict
from fine_tune_util import EvalSampleDatasetTrainer, compute_metrics, preprocess_logits_for_metrics_mlm, token_length_histogram, save_dicts_to_csv, save_metrics
from datetime import datetime

def main(args):

    #current_dir = os.path.dirname(os.path.abspath(__file__))

    model_name = "FacebookAI_roberta-large" 
    model_path = f"src/Local Models/{model_name}"                                              #                                                      
                                                                                            #
    from_pretrained_params_dict = {                                                           #
        "pretrained_model_name_or_path" : model_path,                                                           #
        "device_map":"cuda:0",                                                                  #
        #"torch_dtype": torch.float16                                                          #
    }                                                                                         #
                                                                                            #
    '''lora_config_params_dict = {                                                               #
        "lora_alpha":1024,                                                                      #
        "lora_dropout":0.1,                                                                   #
        "r":512,                                                                               #
        "bias":"none",                                                                        #
        "task_type":TaskType.,                                                       #
    }     '''                                                                                    #
                                                                                            #
    quantization_params_dict = { }                                                                                         #
                                                                                            #
    tokenizer_params_dict = {                                                                 #
    "truncation":True,
    "padding": True,
    "max_length":384                                                                       #
    }                                                                                         #
                                                                                            #
    cur_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")                               #
    checkpoint_dir = f"fine_tuned_llms/{model_name}/checkpoints/{cur_datetime}"        #
    metrics_dir = f"{checkpoint_dir}/metrics.json"                                              #                                                                                          
                                                                                            #
    training_args_dict = {                                                                    #                 
    "output_dir":checkpoint_dir,                                                            #                              
    "per_device_train_batch_size":96, # using A100 gpu, not sure if rivanna can handle more,  #                                                             
    "per_device_eval_batch_size":96,                                                         #                                 
    "num_train_epochs":5,       
    "evaluation_strategy":"steps",                                                          #                                
    "save_strategy":"best",                                                                #                          
    "eval_steps":150,                                                                        #                  
    "save_steps":150,                                                                     #                                
    "metric_for_best_model":"perplexity",                                                                                       
    "greater_is_better":False,              # Lower perplexity is better                    #                                                                      
    "logging_dir":metrics_dir,   
    "logging_strategy": "steps" , 
    "logging_steps": 150,                                                         #                            
    "fp16":True, 
    "learning_rate": 1e-4,    
    "lr_scheduler_type":'constant',
    "eval_on_start": True,     
    "save_safetensors":False,
    "warmup_steps": 100,
    "save_total_limit": 3, # only keeping best 3   
    #"ddp_backend": "nccl",     
    # "torch_compile":True,                                                                 #                                                         #                                            
    }                                                                                         # 
    ###########################################################################################

    #load dataset

    seed = 210
    data = pd.read_csv(args.data_path)  
    #comments = data["comment"].astype(str).sample(frac=0.1, random_state=seed)

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

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast = True)
    model = AutoModelForMaskedLM.from_pretrained(**from_pretrained_params_dict)

    def tokenize_function(examples):
        return tokenizer(examples["text"], **tokenizer_params_dict) 

    # Tokenize each split and remove the 'text' column
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Add 'labels' field for language modeling
    tokenized_datasets = tokenized_datasets.remove_columns(["__index_level_0__"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability = .15)

    # set up trainer
    training_args = TrainingArguments(**training_args_dict)
    trainer = EvalSampleDatasetTrainer(
        eval_sample_size_proportion = .25,
        seed = seed,
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics_mlm,
    )

    trainer.train()

    save_metrics(trainer.state.log_history, f'{checkpoint_dir}/metrics.json')

    hyperparams = {
        "from_pretrained_params": from_pretrained_params_dict,
        #"lora_config_params":lora_config_params_dict,
        "quantization_params":quantization_params_dict,
        "tokenizer_params":tokenizer_params_dict,
        "training_args":training_args_dict
    }


    save_dicts_to_csv(hyperparams, 'fine_tuned_llms/FacebookAI_roberta-large/runs.csv',
    model_name, cur_datetime)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Fine Tune Bert Slurm')

    parser.add_argument('-d', '--data_path',   help='path to dataset', required=True)
    args = parser.parse_args()

    main(args)



