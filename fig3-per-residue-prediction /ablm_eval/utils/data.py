import os
from typing import Union

from datasets import Dataset, DatasetDict, load_dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

__all__ = ["load_and_tokenize"]


def load_and_tokenize(
    data_path: Union[str, dict], tokenizer: PreTrainedTokenizerBase, config
) -> Union[Dataset, DatasetDict]:

    # convert str to dict to simply logic
    return_dataset = False
    if isinstance(data_path, str):
        data_path = {"train": data_path}
        return_dataset = True

    key = next(iter(data_path))  # get name of first Dataset in DatasetDict

    # load
    file_type = os.path.splitext(data_path[key])[1][1:]
    dataset = load_dataset(
        file_type,
        data_files=data_path,
        num_proc=config.num_proc,
    )

    # format sequence column if needed
    columns = dataset[key].column_names
    dataset = _generate_sequence(dataset, column_names=columns, config=config)

    # determine columns to drop
    drop_cols = [col for col in columns if col not in config.keep_columns]

    # tokenize
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(
            x["sequence"],
            padding=config.padding,
            max_length=config.max_len,
            truncation=config.truncate,
            add_special_tokens=config.add_special_tokens,
        ),
        batched=True,
        num_proc=config.num_proc,
        remove_columns=drop_cols,
    )

    # will return Dataset (not DatasetDict) if original path was a string
    return tokenized_dataset[key] if return_dataset else tokenized_dataset


def _generate_sequence(dataset, column_names, config):

    # extract column and separator info
    dataset_columns = config.dataset_columns
    n_chains = len(dataset_columns.chain_names)
    chain_columns = dataset_columns.chain_columns
    separator = config.separator

    for col in chain_columns:
        if col not in column_names:
            raise ValueError(f"The column {col} must exist in the dataset.")

    if n_chains == 1:
        dataset = dataset.map(lambda x: {"sequence": x[chain_columns[0]]})
    else:

        def concat_chains(x):
            seqs = [x[col] for col in chain_columns]
            return {"sequence": separator.join(seqs)}

        # concat chains with separator
        dataset = dataset.map(concat_chains)

    return dataset
