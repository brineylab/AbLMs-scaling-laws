import argparse
import pathlib
from accelerate import Accelerator

from datasets import (
    DatasetDict,
    ClassLabel,
    load_dataset,
)
from transformers import (
    EsmTokenizer,
    EsmForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score
)
import evaluate
accuracy = evaluate.load("accuracy")
from datetime import date
from random import randint
import wandb
import pandas as pd
import numpy as np
import os

# parser
def parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model",
        default=None,
        required=True,
        type=pathlib.Path,
    )
    parser.add_argument(
        "--model_name",
        default=None,
        required=True,
        type=str,
    )
    args = parser.parse_args()
    return args

# processing and tokenization
def preprocess_dataset(
    batch, 
    tokenizer=None, 
    tokenizer_path="facebook/esm2_t12_35M_UR50D", 
    separator="<sep>",
    max_len=320
) -> list:
        
    sequences = [h + separator + l for h, l in zip(batch["h_sequence"], batch["l_sequence"])]
    tokenized = tokenizer(sequences, padding="max_length", max_length=max_len, truncation=True)
    batch["input_ids"] = tokenized.input_ids
    batch["attention_mask"] = tokenized.attention_mask
    
    return batch

# setup training args
def def_training_args(run_name, batch_size=32, lr=5e-5):
    training_args = TrainingArguments(
        run_name = run_name,
        seed=42,
        fp16=True,
        
        # train
        learning_rate=lr,
        per_device_train_batch_size=batch_size, 
        num_train_epochs=3,
        warmup_ratio=0.1,
        lr_scheduler_type='linear',
        
        # eval
        eval_strategy = "steps",
        eval_steps=250,
        per_device_eval_batch_size=batch_size, 
        eval_accumulation_steps=50,

        # saving & logging
        logging_steps=50,
        save_strategy="no",
        output_dir=f"./checkpoints/{run_name}",
        report_to="wandb",
        logging_dir=f"./logs/{run_name}",
        logging_first_step=True,
        
    )
    return training_args

# classification metrics
# for attention classifier
def compute_metrics_atten(eval_pred):
    logits, labels = eval_pred
    predictions = logits[0] # logits[0] are predictions, logits[1] is classifier attention
    probabilities = torch.softmax(torch.from_numpy(predictions), dim=1).detach().numpy()[:,-1]
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "precision": precision_score(labels, predictions, pos_label=1),
        "recall": recall_score(labels, predictions, pos_label=1),
        "f1": f1_score(labels, predictions, pos_label=1),
        "auc": roc_auc_score(labels, probabilities),
        "aupr": average_precision_score(labels, probabilities, pos_label=1),
        "mcc": matthews_corrcoef(labels, predictions),
    }

# classification metrics
# for normal classifier
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    probabilities = torch.softmax(torch.from_numpy(predictions), dim=1).detach().numpy()[:,-1]
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "precision": precision_score(labels, predictions, pos_label=1),
        "recall": recall_score(labels, predictions, pos_label=1),
        "f1": f1_score(labels, predictions, pos_label=1),
        "auc": roc_auc_score(labels, probabilities),
        "aupr": average_precision_score(labels, probabilities, pos_label=1),
        "mcc": matthews_corrcoef(labels, predictions),
    }

def main():
    # args
    args = parser()

    # labels
    class_labels = ClassLabel(names=['Healthy-donor','Sars-specific'])
    n_classes = len(class_labels.names)
    label2id = {"Healthy-donor": 0, "Sars-specific": 1}
    id2label = {0: "Healthy-donor", 1: "Sars-specific"}

    # tokenizer
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")

    # test results
    results = pd.DataFrame({"model": [],
                            "itr": [],
                            "test_loss": [],
                            "test_accuracy": [],
                            "test_precision": [],
                            "test_recall": [],
                            "test_f1": [],
                            "test_auc": [],
                            "test_aupr": [],
                            "test_mcc": [],
                            })

    # wandb
    os.environ['WANDB_PROJECT'] = 'Class_CoV_b128_5fold_FINAL'
    os.environ['WANDB_RUN_GROUP'] = 'esm_150M_full_b128_HD-CoV'
    os.environ['WANDB_JOB_TYPE'] = args.model_name
    
    # loop through datasets
    for i in range(5):

        run_name = f"{args.model_name}_HD-CoV-paired-class_itr{i}_{date.today().isoformat()}"
        training_args = def_training_args(run_name)

        # load dataset
        data_files = DatasetDict({
            'train': f'/home/jovyan/shared/mahdi/1_projects/model_optimization/02classification/data/5_folded/hd-0_cov-1_train{i}.csv',
            'test': f'/home/jovyan/shared/mahdi/1_projects/model_optimization/02classification/data/5_folded/hd-0_cov-1_test{i}.csv'
        })
        dataset = load_dataset('csv', data_files=data_files)

        # tokenize
        tokenized_dataset = dataset.map(
            preprocess_dataset,
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_len": 320,
            },
            batched=True,
            remove_columns=["name", "h_sequence", "l_sequence"]
        )
    
        # model
        model = EsmForSequenceClassification.from_pretrained(
        #model = EsmForSequenceClassification.from_pretrained(
            args.model, 
            num_labels=n_classes,
            label2id=label2id,
            id2label=id2label,
        )
        for param in model.base_model.parameters():
            param.requires_grad = False
    
        # trainer
        trainer = Trainer(
            model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['test'],
            compute_metrics=compute_metrics
        )
        trainer.train()
        
    
        # eval
        logits, labels, metrics = trainer.predict(tokenized_dataset['test'])

        # end
        wandb.finish()
        del model
    
        # save
        filtered_metrics = {key: value for key, value in metrics.items() if key in results.columns}
        filtered_metrics['model'] = args.model_name
        filtered_metrics['itr'] = i
        results = results.append(filtered_metrics, ignore_index=True)

    results.to_csv(f"/home/jovyan/shared/mahdi/1_projects/model_optimization/02classification/results/{args.model_name}_HD-CoV-5fold_results.csv", index=False)

if __name__ == "__main__":
    main()
