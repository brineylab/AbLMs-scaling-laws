import os
import warnings
warnings.simplefilter('ignore')
from transformers import (
    EsmConfig, 
    EsmTokenizer, 
    EsmForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
import torch
from datasets import load_dataset
import argparse
import pathlib
import wandb
from datetime import date
import yaml

def parser():
    parser = argparse.ArgumentParser()
    
    # train argument parser
    parser.add_argument(
        "--train_config",
        default=None,
        required=True,
        type=pathlib.Path,
        help="yaml file containing training arguments is required!",
    )
    
    args = parser.parse_args()
    return args

# this config setup is specific for the folding model
def define_config(train_config, tokenizer):

    config = EsmConfig(
        vocab_size=train_config.get("vocab_size", 26),
        pad_token_id=train_config.get("pad_token_id", 21),
        mask_token_id=train_config.get("mask_token_id", 22),
        num_attention_heads=train_config.get("num_attention_heads", 20),
        num_hidden_layers=train_config.get("num_hidden_layers", 32),
        hidden_size=train_config.get("hidden_size", 960),
        intermediate_size=train_config.get("intermediate_size", 3840),
        max_position_embeddings=train_config.get("max_position_embeddings", 322),
        position_embedding_type=train_config.get("position_embedding_type", "rotary"),
    )
    return config


def define_args(train_config, run_name):
    
    # setup training arguments
    training_args = TrainingArguments(
        run_name=run_name,
        fp16=train_config.get("fp16", True),
        seed=train_config.get("seed", 42),

        # batch sizes
        per_device_train_batch_size=train_config.get("batch_size", 32),
        per_device_eval_batch_size=train_config.get("batch_size", 32),
        
        # eval
        evaluation_strategy=train_config.get("evaluation_strategy", "steps"),
        eval_steps=train_config.get("eval_steps", 25000),
        
        # training
        max_steps=train_config.get("max_steps", 500000),
        save_steps=train_config.get("save_steps", 50000),
        adam_beta1=train_config.get("adam_beta1", 0.9),
        adam_beta2=train_config.get("adam_beta2", 0.98),
        adam_epsilon=train_config.get("adam_epsilon", 1e-6),
        weight_decay=train_config.get("weight_decay", 0.01),
        warmup_steps=train_config.get("warmup_steps", 30000),
        learning_rate=train_config.get("peak_learning_rate", 4e-4),
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 1),

        # output and logging
        logging_steps=train_config.get("logging_steps", 100),
        output_dir=train_config.get("output_dir").format(run_name=run_name),
        overwrite_output_dir=train_config.get("overwrite_output_dir", True),
        logging_dir=train_config.get("logging_dir", f"../wandb/{run_name}"),
        report_to=train_config.get("report_to", None),
        logging_first_step=train_config.get("logging_first_step", True),
    )
    
    return training_args



def preprocess_dataset(
    seq, 
    tokenizer, 
    separator,
    padding="max_length",
    truncation=True,
    max_len=512
) -> list:
        
    # reformat sequences with sep tokens
    sequence = seq['sequence_aa_heavy'] + separator + seq['sequence_aa_light']
    
    # tokenize
    tokenized = tokenizer(sequence, 
                          padding=padding, 
                          max_length=max_len,
                          truncation=truncation)
    
    # special tokens mask - after tokenizer
    # so <cls> seperator tokens get added to attention mask
    tokenized['special_tokens_mask'] = tokenizer.get_special_tokens_mask(tokenized['input_ids'], 
                                    already_has_special_tokens=True)
    
    return tokenized


def load_and_tokenize(train_config, tokenizer):
    data_files = {
        "train": [train_config.get("train_file")],
        "eval": [train_config.get("validation_file")],
    }
    dataset = load_dataset(train_config.get("file_type"), data_files=data_files)
#dataset is a DatasetDict object containing the loaded datasets for the specified splits.
    tokenized_dataset = dataset.map(
        preprocess_dataset,
        fn_kwargs={
            "tokenizer": tokenizer,
            "padding": train_config.get("padding"),
            "max_len": train_config.get("max_length"),
            "truncation": train_config.get("truncation"),
            "separator": train_config.get("separator_token")
        },
        num_proc=100,
        #batched=True, # note this function isn't written to support batching!!
        remove_columns=["sequence"]
    )
    return tokenized_dataset

def main():
    # parse cl args
    args = parser()
    with open(args.train_config, 'r') as stream:
        train_config = yaml.safe_load(stream)

    run_name = train_config.get("run_name") + f"_{date.today().isoformat()}"

    # tokenize
    tokenizer = EsmTokenizer.from_pretrained(train_config.get("tokenizer_path"))
    tokenized_dataset = load_and_tokenize(train_config, tokenizer)

    # define model config
    model_config = define_config(train_config, tokenizer)

    # define training args
    training_args = define_args(train_config, run_name)
    
    # collator
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=train_config.get("mlm", True), 
        mlm_probability=train_config.get("mlm_probability", 0.15)
    )
    
    # wandb
    # don't call wandb.init() -> let Trainer call it automatically,
    # otherwise multiple runs will be initilized
    wandb.login()
    os.environ["WANDB_PROJECT"] = train_config.get("wandb_project")
    
    # model
    model = EsmForMaskedLM(model_config)

    model_size = sum(p.numel() for p in model.parameters())
    print(f"Model size: {model_size/1e6:.2f}M")
    
    # train
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["eval"]
    )
    trainer.train()
    
    # save and end
    trainer.save_model(f"./models/{run_name}")
    
    # don't finish wandb manually - it will finish when the script ends
    
if __name__ == "__main__":
    main()
