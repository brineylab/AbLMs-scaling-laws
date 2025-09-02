import gc
import json
import pathlib
from dataclasses import asdict

import torch


__all__ = ["evaluate_ablms"]

BOLD = "\033[1m"
UNDERLINE = "\033[4m"
RESET = "\033[0m"


def evaluate_ablms(
    models: dict,
    configs: list,
    shared_output_dir: str = "./results",
    ignore_existing_files: bool = False,
):
    """
    Evaluate the provided models on the
    given tasks.
    """

    # create output dirs
    create_results_dir(shared_output_dir, configs, ignore_existing_files)

    # eval
    for itr, (model_name, model_path) in enumerate(models.items(), 1):
        print(f"\n{BOLD}Evaluating Model #{itr}: {model_name}{RESET}")
        _eval_model(model_name, model_path, configs)


def create_results_dir(output_dir: str, configs: list, ignore_existing: bool):
    """
    Create directory structure for results.
    """
    # base directory
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # task directories
    for config in configs:
        # task dir path
        task_path = output_path / f"{config.task_dir}"
        config.output_dir = str(task_path)

        # make task dir
        if not ignore_existing:
            _check_dir(task_path)
        task_path.mkdir(exist_ok=True)

        # results dir inside task dir
        subdir_path = task_path / "results"
        subdir_path.mkdir(exist_ok=True)

        # save config
        with open(f"{task_path}/config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)


def _check_dir(path: pathlib.Path):
    """
    Raise exception if the given directory or any of its subdirectories contain files.
    """
    if path.exists() and any(p.is_file() for p in path.rglob("*")):
        raise Exception(f"The directory '{path}' exists and is not empty!")


def _eval_model(model_name: str, model_path: str, configs: list):
    for itr, config in enumerate(configs, 1):
        # run task
        print(f"{UNDERLINE}Running Task #{itr}: {config.name}{RESET}")
        task_fn = config.runner
        task_fn(model_name, model_path, config)

        # clean up memory
        _clean_up()


def _clean_up():
    gc.collect()
    torch.cuda.empty_cache()
