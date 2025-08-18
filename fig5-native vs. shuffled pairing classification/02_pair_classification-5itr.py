"""Binary Classification Model Head Training (Fine-tuning) - Native vs Shuffled Pairs"""

# data loading and cleaning
import pandas as pd
import numpy as np
from datasets import (
    DatasetDict,
    ClassLabel,
    load_dataset,
)

# model and training
from transformers import (
    EsmTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# metrics
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
)
import evaluate

accuracy = evaluate.load("accuracy")

from datetime import date
import os
import argparse
import pathlib
import yaml
import torch
import wandb


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
        eval_steps=train_config.get("eval_steps", 500),
        # training
        num_train_epochs=train_config.get("num_train_epochs", 1),
        learning_rate=train_config.get("learning_rate", 5e-5),
        lr_scheduler_type=train_config.get("lr_scheduler_type", "linear"),
        warmup_ratio=train_config.get("warmup_ratio", 0.1),
        save_strategy=train_config.get("save_strategy", "no"),
        # output and logging
        logging_steps=train_config.get("logging_steps", 100),
        logging_dir=train_config.get("logging_dir", f"./logs/") + run_name,
        logging_first_step=train_config.get("logging_first_step", True),
        output_dir=train_config.get("output_dir", f"./output/") + run_name,
        report_to=train_config.get("report_to", "none"),
    )

    return training_args


def preprocess_dataset(batch, tokenizer, separator="<cls><cls>", max_len=320) -> list:

    # tokenize the H/L sequence pair
    sequences = [
        h + separator + l for h, l in zip(batch["h_sequence"], batch["l_sequence"])
    ]
    tokenized = tokenizer(sequences, padding="max_length", max_length=max_len)
    batch["input_ids"] = tokenized.input_ids
    batch["attention_mask"] = tokenized.attention_mask

    return batch


def load_and_tokenize(train_config, tokenizer, class_labels):

    # read in 5 data splits
    itr_datasets = []
    # for i in range(5):
    for i in range(5):
        data_files = DatasetDict(
            {
                "train": f'{train_config.get("data_path")}train{i}.csv',
                "test": f'{train_config.get("data_path")}test{i}.csv',
            }
        )
        dataset = load_dataset("csv", data_files=data_files)
        itr_datasets.append(dataset)

    # tokenize all splits
    tokenized = []
    for dataset in itr_datasets:
        tokenized_dataset = dataset.map(
            preprocess_dataset,
            fn_kwargs={
                "tokenizer": tokenizer,
                "separator": train_config.get("separator_token"),
                "max_len": 320,
            },
            batched=True,
            remove_columns=["h_sequence", "l_sequence", "donor"],
        )
        tokenized.append(tokenized_dataset)

    return tokenized


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    probabilities = (
        torch.softmax(torch.from_numpy(predictions), dim=1).detach().numpy()[:, -1]
    )
    predictions = np.argmax(predictions, axis=1)
    _accuracy = accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    return {
        "accuracy": _accuracy,
        "precision": precision_score(labels, predictions, pos_label=1),
        "recall": recall_score(labels, predictions, pos_label=1),
        "f1": f1_score(labels, predictions, pos_label=1),
        "auc": roc_auc_score(labels, probabilities),
        "aupr": average_precision_score(labels, probabilities, pos_label=1),
        "mcc": matthews_corrcoef(labels, predictions),
    }


def main():
    # parse cl args
    args = parser()
    with open(args.train_config, "r") as stream:
        train_config = yaml.safe_load(stream)

    # class labels
    class_labels = ClassLabel(
        names=[train_config.get("class_0"), train_config.get("class_1")]
    )
    n_classes = len(class_labels.names)
    label2id = {train_config.get("class_0"): 0, train_config.get("class_1"): 1}
    id2label = {0: train_config.get("class_0"), 1: train_config.get("class_1")}

    # tokenize
    tokenizer = EsmTokenizer.from_pretrained(train_config.get("tokenizer_path"))
    tokenized_dataset = load_and_tokenize(train_config, tokenizer, class_labels)

    # base model
    model_id = train_config.get("model_name")
    model_path = train_config.get("model_path")

    # run statistics
    test_results = pd.DataFrame(
        {
            "model_name": [],
            "itr": [],
            "test_loss": [],
            "test_accuracy": [],
            "test_precision": [],
            "test_recall": [],
            "test_f1": [],
            "test_auc": [],
            "test_aupr": [],
            "test_mcc": [],
        }
    )

    for n, dataset in enumerate(tokenized_dataset):

        # run name (to include base model, classification task name, and date)
        run_name = f"{model_id}_itr{n}_{train_config.get('task_name')}_{date.today().isoformat()}"
        print(run_name)

        # model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=n_classes,
            label2id=label2id,
            id2label=id2label,
        )

        # freeze base model weights (as opposed to fine tuning)
        if train_config.get("freeze_weights"):
            for param in model.base_model.parameters():
                param.requires_grad = False

        # define training args
        training_args = define_args(train_config, run_name)

        # wandb (don't call wandb.init() -> let Trainer call it automatically, otherwise multiple runs will be initilized)
        if train_config.get("report_to") == "wandb":
            os.environ["WANDB_PROJECT"] = train_config.get("wandb_project")
            os.environ["WANDB_RUN_GROUP"] = train_config.get("wandb_group")
            os.environ["WANDB_JOB_TYPE"] = train_config.get("model_name")
        wandb.login()
        # train
        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=compute_metrics,
        )
        trainer.train()

        # save and end
        if train_config.get("save_model"):
            trainer.save_model(
                train_config.get("model_save_dir", f"./models/")
                + training_args.run_name
            )

        # evaluate
        logits, labels, metrics = trainer.predict(dataset["test"])
        metrics["model_name"] = train_config.get("model_name")
        metrics["itr"] = n
        test_results.loc[len(test_results)] = metrics
        print(metrics)

        wandb.finish()  # necessary since script does not end immediately after model is done training

        # delete to ensure untrained head is being trained for each base model
        del model

    # save run stats to csv
    print(test_results)
    test_results.to_csv(
        train_config.get("results_dir", f"./results/")
        + f"{model_id}_itr-tests_{train_config.get('task_name')}.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
