# evaluate_cli.py
import argparse
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

from datasets import Dataset as HFDataset

from .api import _source_to_dataset, evaluate, load_recipe
from .artifact import UnitxtArtifactNotFoundError
from .benchmark import Benchmark

# Use HFAutoModelInferenceEngine for local models
from .inference import (
    CrossProviderInferenceEngine,
    HFAutoModelInferenceEngine,
    InferenceEngine,
)
from .logging_utils import get_logger
from .metric_utils import EvaluationResults
from .parsing_utils import parse_key_equals_value_string_to_dict
from .settings_utils import settings

# Define logger early so it can be used in initial error handling
# Basic config for initial messages, will be reconfigured in main()
logger = get_logger()


def try_parse_json(value: str) -> Union[str, dict, None]:
    """Attempts to parse a string as JSON or key=value pairs.

    Returns the original string if parsing fails
    and the string doesn't look like JSON/kv pairs.
    Raises ArgumentTypeError if it looks like JSON but is invalid.
    """
    if value is None:
        return None
    try:
        # Handle simple key-value pairs like "key=value,key2=value2"
        if "=" in value and "{" not in value:
            parsed_dict = parse_key_equals_value_string_to_dict(value)
            if parsed_dict:
                return parsed_dict

        # Attempt standard JSON parsing
        return json.loads(value)

    except json.JSONDecodeError as e:
        if value.strip().startswith("{") or value.strip().startswith("["):
            raise argparse.ArgumentTypeError(
                f"Invalid JSON: '{value}'. Hint: Use double quotes for JSON strings and check syntax."
            ) from e
        return value  # Return as string if not JSON-like
    except Exception as e:
        logger.error(f"Error parsing argument '{value}': {e}")
        raise argparse.ArgumentTypeError(f"Could not parse argument: '{value}'") from e


def setup_parser() -> argparse.ArgumentParser:
    """Sets up the argument parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="CLI utility for running evaluations with unitxt.",
    )

    # --- Task/Dataset Arguments ---
    parser.add_argument(
        "--tasks",  # Changed to plural to better reflect it holds a list
        "-t",
        dest="tasks",  # Explicitly set the attribute name to 'tasks'
        type=partial(str.split, sep="+"),  # Use the custom function for type conversion
        required=True,
        help="Plus-separated (+) list of Unitxt task/dataset identifier strings.\n"
        "Each task format: 'card=<card_ref>,template=<template_ref>,...'\n"
        "Example: 'card=cards.mmlu,t=t.mmlu.all+card=cards.hellaswag,t=t.hellaswag.no'",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (e.g., 'train', 'validation', 'test'). Default: 'test'.",
    )
    parser.add_argument(
        "--num_fewshots",
        type=int,
        default=None,
        help="number of fewshots to use",
    )
    parser.add_argument(
        "--limit",
        "-L",
        type=int,
        default=None,
        metavar="N",
        help="Limit the number of examples per task/dataset.",
    )

    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=1,
        help="Batch size for use in inference when selected model is hf. Default 1",
    )

    # --- Model Arguments (Explicit Types) ---
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="hf",
        choices=["hf", "cross_provider"],
        help="Specifies the model type/engine.\n"
        "- 'hf': Local Hugging Face model via HFAutoModel (default). Requires 'pretrained=...' in --model_args.\n"
        "- 'cross_provider': Remote model via CrossProviderInferenceEngine. Requires 'model_name=...' in --model_args.",
    )
    parser.add_argument(
        "--model_args",
        "-a",
        type=try_parse_json,
        default={},
        help="Comma separated string or JSON formatted arguments for the model/inference engine.\n"
        "Examples:\n"
        "- For --model hf (default): 'pretrained=meta-llama/Llama-3.1-8B-Instruct,torch_dtype=bfloat16,device=cuda'\n"
        "  (Note: 'pretrained' key is REQUIRED. Other args like 'torch_dtype', 'device', generation params are passed too)\n"
        "- For --model generic_remote: 'model_name=llama-3-3-70b-instruct,max_tokens=256,temperature=0.7'\n"
        "  (Note: 'model_name' key is REQUIRED)\n"
        '- JSON format: \'{"pretrained": "my_model", "torch_dtype": "float32"}\' or \'{"model_name": "openai/gpt-4o"}\'',
    )

    parser.add_argument(
        "--gen_kwargs",
        type=try_parse_json,
        default=None,
        help=(
            "Comma delimited string for model generation on greedy_until tasks,"
            """ e.g. temperature=0,top_p=0.1."""
        ),
    )

    parser.add_argument(
        "--chat_template_kwargs",
        type=try_parse_json,
        default=None,
        help=(
            "Comma delimited string for tokenizer kwargs"
            "e.g. thinking=True (https://github.com/huggingface/transformers/blob/9a1c1fe7edaefdb25ab37116a979832df298d6ea/src/transformers/tokenization_utils_base.py#L1542)"
        ),
    )

    # --- Output and Logging Arguments ---
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        default=".",
        help="Directory to save evaluation results and logs. Default: current directory.",
    )
    parser.add_argument(
        "--output_file_prefix",
        type=str,
        default="evaluation_results",
        help="Prefix for the output JSON file names. Default: 'evaluation_results'.",
    )
    parser.add_argument(
        "--log_samples",
        "-s",
        action="store_true",
        default=False,
        help="If True, save individual predictions and scores to a separate JSON file.",
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        type=str.upper,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Controls logging verbosity level. Default: INFO.",
    )

    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        default=False,
    )

    # --- Unitxt Settings ---
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=False,
        help="Allow execution of unverified code from the HuggingFace Hub (used by datasets/unitxt).",
    )
    parser.add_argument(
        "--disable_hf_cache",
        action="store_true",
        default=False,
        help="Disable HuggingFace datasets caching.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory for HuggingFace datasets cache (overrides default).",
    )

    return parser


def setup_logging(verbosity: str) -> None:
    """Configures logging based on verbosity level."""
    logging.basicConfig(
        level=verbosity,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,  # Ensures reconfiguration works if basicConfig was called before
    )
    # Re-get the logger instance after basicConfig is set
    global logger
    logger = get_logger()
    logger.setLevel(verbosity)


def prepare_output_paths(output_path: str, prefix: str) -> Tuple[str, str]:
    """Creates output directory and defines file paths.

    Args:
        output_path (str): The directory where output files will be saved.
        prefix (str): The prefix for the output file names.

    Returns:
        Tuple[str, str]: A tuple containing the path for the results summary file
                         and the path for the detailed samples file.
    """
    os.makedirs(output_path, exist_ok=True)
    results_file_path = os.path.join(output_path, f"{prefix}.json")
    samples_file_path = os.path.join(output_path, f"{prefix}_samples.json")
    return results_file_path, samples_file_path


def configure_unitxt_settings(args: argparse.Namespace):
    """Configures unitxt settings and returns a context manager.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        ContextManager: A context manager for applying unitxt settings.
    """
    unitxt_settings_dict = {
        "disable_hf_datasets_cache": args.disable_hf_cache,
        "allow_unverified_code": args.trust_remote_code,
    }
    if args.cache_dir:
        unitxt_settings_dict["hf_cache_dir"] = args.cache_dir
        # Also set environment variable as some HF parts might read it directly
        os.environ["HF_DATASETS_CACHE"] = args.cache_dir
        os.environ["HF_HOME"] = args.cache_dir
        logger.info(f"Set HF_DATASETS_CACHE to: {args.cache_dir}")

    if args.disable_hf_cache:
        os.environ["UNITXT_DISABLE_HF_DATASETS_CACHE"] = "True"

    logger.info(f"Applying unitxt settings: {unitxt_settings_dict}")
    return settings.context(**unitxt_settings_dict)


def cli_load_dataset(args: argparse.Namespace) -> HFDataset:
    """Loads the dataset based on command line arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        HFDataset: The loaded dataset.

    Raises:
        UnitxtArtifactNotFoundError: If the specified card or template artifact is not found.
        FileNotFoundError: If a specified file (e.g., in a local card path) is not found.
        AttributeError: If there's an issue accessing attributes during loading.
        ValueError: If there's a value-related error during loading (e.g., parsing).
    """
    logger.info(
        f"Loading task/dataset using identifier: '{args.tasks}' with split '{args.split}'"
    )

    benchmark_subsets = {}
    for task_str in args.tasks:
        overwrite_args = extract_overwrite_args(args)
        benchmark_subsets[task_str] = load_recipe(
            dataset_query=task_str, **overwrite_args
        )

    # this hack circumvents an issue with multi-level benchmarks (such Bluebench's translation subset) that fail when wrapped with an additional Benchmark() object.
    if len(benchmark_subsets) == 1:
        source = next(iter(benchmark_subsets.values()))
    else:
        source = Benchmark(subsets=benchmark_subsets)

    test_dataset = _source_to_dataset(source, split=args.split)
    logger.info(
        f"Dataset loaded successfully. Number of instances: {len(test_dataset)}"
    )
    return test_dataset


def extract_overwrite_args(args):
    dataset_args = {}

    if args.limit is not None:
        assert (
            f"max_{args.split}_instances" not in dataset_args
        ), "limit was inputted both as an arg and as a task parameter"
        # Check if limit or loader_limit is already present
        # dataset_args[f"max_{args.split}_instances"] = args.limit
        dataset_args[f"max_{args.split}_instances"] = args.limit
        # Use loader_limit for unitxt compatibility
        logger.info(
            f"Applying limit from --limit argument: max_{args.split}_instances={args.limit}"
        )

    if args.num_fewshots:
        assert (
            "num_demos" not in dataset_args
        ), "num_demos was inputted both as an arg and as a task parameter"
        dataset_args["num_demos"] = args.num_fewshots
        dataset_args.update(
            {
                "demos_taken_from": "train",
                "demos_pool_size": -1,
                "demos_removed_from_data": True,
            }
        )  # Use loader_limit for unitxt compatibility
        logger.info(
            f"Applying limit from --limit argument: num_demos={args.num_fewshots}"
        )

    if args.apply_chat_template:
        assert (
            "format" not in dataset_args
        ), "format was inputted as a task parameter, but chat_api was requested"
        dataset_args["format"] = "formats.chat_api"
        logger.info(
            "Applying chat template from --apply_chat_template argument: format=formats.chat_api"
        )

    return dataset_args


def prepare_kwargs(kwargs: dict) -> Dict[str, Any]:
    """Prepares the model arguments dictionary.

    Args:
        kwargs (dict): Parsed command-line arguments.

    Returns:
        Dict[str, Any]: The processed model arguments dictionary.
    """
    # Ensure model_args is a dictionary, handling potential string return from try_parse_json
    kwargs_dict = kwargs if isinstance(kwargs, dict) else {}
    if not isinstance(kwargs, dict) and kwargs is not None:
        logger.warning(
            f"Could not parse kwargs '{kwargs}' as JSON or key-value pairs. Treating as empty."
        )

    logger.info(f"Using kwargs: {kwargs_dict}")
    return kwargs_dict


def initialize_inference_engine(
    args: argparse.Namespace,
    model_args_dict: Dict[str, Any],
    chat_kwargs_dict: Dict[str, Any],
) -> InferenceEngine:
    """Initializes the appropriate inference engine based on arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        model_args_dict (Dict[str, Any]): Processed model arguments.
        chat_kwargs_dict (Dict[str, Any]): Processed chat arguments.

    Returns:
        InferenceEngine: The initialized inference engine instance.

    Raises:
        SystemExit: If required dependencies are missing for the selected model type.
        ValueError: If required keys are missing in model_args for the selected model type.
    """
    inference_model = None
    # --- Local Hugging Face Model (using HFAutoModelInferenceEngine) ---
    if args.model.lower() == "hf":
        if "pretrained" not in model_args_dict:
            logger.error(
                "Missing 'pretrained=<model_id_or_path>' in --model_args for '--model hf'."
            )
            raise ValueError(
                "Argument 'pretrained' is required in --model_args when --model is 'hf'"
            )

        local_model_name = model_args_dict.pop("pretrained")
        logger.info(
            f"Initializing HFAutoModelInferenceEngine for model: {local_model_name}"
        )

        model_args_dict.update({"batch_size": args.batch_size})
        logger.info(f"HFAutoModelInferenceEngine args: {model_args_dict}")

        inference_model = HFAutoModelInferenceEngine(
            model_name=local_model_name,
            **model_args_dict,
            chat_kwargs_dict=chat_kwargs_dict,
        )

        # Keep the actual model name for the results
        args.model = inference_model.model_name
    # --- Remote Model (CrossProviderInferenceEngine) ---
    elif args.model.lower() == "cross_provider":
        if "model_name" not in model_args_dict:
            logger.error(
                "Missing 'model_name=<provider/model_id>' in --model_args for '--model cross_provider'."
            )
            raise ValueError(
                "Argument 'model_name' is required in --model_args when --model is 'cross_provider'"
            )

        remote_model_name = model_args_dict.pop("model_name")
        logger.info(
            f"Initializing CrossProviderInferenceEngine for model: {remote_model_name}"
        )

        if (
            "max_tokens" not in model_args_dict
            and "max_new_tokens" not in model_args_dict
        ):
            logger.warning(
                f"'max_tokens' or 'max_new_tokens' not found in --model_args, {remote_model_name} might require it."
            )

        logger.info(f"CrossProviderInferenceEngine args: {model_args_dict}")

        # Note: CrossProviderInferenceEngine expects 'model' parameter, not 'model_name'
        inference_model = CrossProviderInferenceEngine(
            model=remote_model_name,
            **model_args_dict,
        )

        # Keep the actual model name for the results
        args.model = inference_model.engine.model
    else:
        # This case should not be reached due to argparse choices
        logger.error(
            f"Invalid --model type specified: {args.model}. Use 'hf' or 'cross_provider'."
        )
        sys.exit(1)  # Exit here as it's an invalid configuration

    return inference_model


def run_inference(engine: InferenceEngine, dataset: HFDataset) -> List[Any]:
    """Runs inference using the initialized engine.

    Args:
        engine (InferenceEngine): The inference engine instance.
        dataset (HFDataset): The dataset to run inference on.

    Returns:
        List[Any]: A list of predictions.

    Raises:
        Exception: If an error occurs during inference.
    """
    logger.info("Starting inference...")
    try:
        predictions = engine.infer(dataset)
        logger.info("Inference completed.")
        if not predictions:
            logger.warning("Inference returned no predictions.")
            return []  # Return empty list if no predictions
        if len(predictions) != len(dataset):
            logger.error(
                f"Inference returned an unexpected number of predictions ({len(predictions)}). Expected {len(dataset)}."
            )
            # Don't exit, but log error. Evaluation might still work partially or fail later.
        return predictions
    except Exception:
        logger.exception("An error occurred during inference")  # Use logger.exception
        raise  # Re-raise after logging


def run_evaluation(predictions: List[Any], dataset: HFDataset) -> EvaluationResults:
    """Runs evaluation on the predictions.

    Args:
        predictions (List[Any]): The list of predictions from the model.
        dataset (HFDataset): The dataset containing references and other data.

    Returns:
        EvaluationResults: The evaluated dataset (list of instances with scores).

    Raises:
        RuntimeError: If evaluation returns no results or an unexpected type.
        Exception: If any other error occurs during evaluation.
    """
    logger.info("Starting evaluation...")
    if not predictions:
        logger.warning("Skipping evaluation as there are no predictions.")
        return []  # Return empty list if no predictions to evaluate

    try:
        evaluation_results = evaluate(predictions=predictions, data=dataset)
        logger.info("Evaluation completed.")
        if not evaluation_results:
            logger.error("Evaluation returned no results (empty list/None).")
            # Raise an error as this indicates a problem in the evaluation process
            raise RuntimeError("Evaluation returned no results.")
        if not isinstance(evaluation_results, EvaluationResults):
            logger.error(
                f"Evaluation returned unexpected type: {type(evaluation_results)}. Expected list."
            )
            raise RuntimeError(
                f"Evaluation returned unexpected type: {type(evaluation_results)}"
            )

        return evaluation_results
    except Exception:
        logger.exception("An error occurred during evaluation")  # Use logger.exception
        raise  # Re-raise after logging


def _get_unitxt_commit_hash() -> Optional[str]:
    """Tries to get the git commit hash of the installed unitxt package."""
    try:
        # Find the directory of the unitxt package
        # Use inspect to be more robust finding the package path

        current_script_path = os.path.abspath(__file__)
        package_dir = os.path.dirname(current_script_path)

        # Check if it's a git repository and get the commit hash
        # Use absolute path for git command
        git_command = ["git", "-C", os.path.abspath(package_dir), "rev-parse", "HEAD"]
        logger.debug(f"Running git command: {' '.join(git_command)}")
        result = subprocess.run(
            git_command,
            capture_output=True,
            text=True,
            check=False,  # Don't raise error if git command fails
            encoding="utf-8",
            errors="ignore",  # Ignore potential decoding errors
        )
        if result.returncode == 0:
            commit_hash = result.stdout.strip()
            logger.info(f"Found unitxt git commit hash: {commit_hash}")
            # Verify it looks like a hash (e.g., 40 hex chars)
            if len(commit_hash) == 40 and all(
                c in "0123456789abcdef" for c in commit_hash
            ):
                return commit_hash
            logger.warning(
                f"Git command output '{commit_hash}' doesn't look like a valid commit hash."
            )
            return None
        stderr_msg = result.stderr.strip() if result.stderr else "No stderr"
        logger.warning(
            f"Could not get unitxt git commit hash (git command failed with code {result.returncode}): {stderr_msg}"
        )
        return None
    except ImportError:
        logger.warning("unitxt package not found, cannot determine commit hash.")
        return None
    except FileNotFoundError:
        logger.warning(
            "'git' command not found in PATH. Cannot determine unitxt commit hash."
        )
        return None
    except Exception as e:
        logger.warning(
            f"Error getting unitxt commit hash: {e}", exc_info=True
        )  # Log traceback
        return None


def _get_installed_packages() -> Dict[str, str]:
    """Gets a dictionary of installed packages and their versions."""
    packages = {}
    try:
        for dist in importlib.metadata.distributions():
            # Handle potential missing metadata gracefully
            name = dist.metadata.get("Name")
            version = dist.metadata.get("Version")
            if name and version:
                packages[name] = version
            elif name:
                packages[name] = "N/A"  # Record package even if version is missing
                logger.debug(f"Could not find version for package: {name}")

        logger.info(f"Collected versions for {len(packages)} installed packages.")
    except Exception as e:
        logger.warning(f"Could not retrieve installed package list: {e}", exc_info=True)
    return packages


def _get_unitxt_version() -> str:
    """Gets the installed unitxt version using importlib.metadata."""
    try:
        version = importlib.metadata.version("unitxt")
        logger.info(f"Found unitxt version using importlib.metadata: {version}")
        return version
    except importlib.metadata.PackageNotFoundError:
        logger.warning(
            "Could not find 'unitxt' package version using importlib.metadata. Is it installed correctly?"
        )
        return "N/A"
    except Exception as e:
        logger.warning(
            f"Error getting unitxt version using importlib.metadata: {e}", exc_info=True
        )
        return "N/A"


def prepend_timestamp_to_path(original_path, timestamp):
    """Takes a path string and a timestamp string, prepends the timestamp to the filename part of the path, and returns the new path string."""
    directory, filename = os.path.split(original_path)
    # Use an f-string to create the new filename with the timestamp prepended
    new_filename = f"{timestamp}_{filename}"
    # Join the directory and the new filename back together
    return os.path.join(directory, new_filename)


def _save_results_to_disk(
    args: argparse.Namespace,
    global_scores: Dict[str, Any],
    all_samples_data: Dict[str, List[Dict[str, Any]]],
    results_path: str,
    samples_path: str,
) -> None:
    """Saves the configuration, environment info, global scores, and samples to JSON files.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        global_scores (Dict[str, Any]): Dictionary of global scores.
        all_samples_data (Dict[str, List[Dict[str, Any]]]): List of processed sample data.
        results_path (str): Path to save the summary results JSON file.
        samples_path (str): Path to save the detailed samples JSON file.
    """
    # --- Gather Configuration ---
    config_to_save = {}
    for k, v in vars(args).items():
        # Ensure complex objects are represented as strings
        if isinstance(v, (str, int, float, bool, list, dict, type(None))):
            config_to_save[k] = v
        else:
            try:
                # Try standard repr first
                config_to_save[k] = repr(v)
            except Exception:
                # Fallback if repr fails
                config_to_save[
                    k
                ] = f"<Object of type {type(v).__name__} could not be represented>"

    # --- Gather Environment Info ---
    unitxt_commit = _get_unitxt_commit_hash()
    # Get version using the dedicated function
    unitxt_pkg_version = _get_unitxt_version()

    environment_info = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "command_line_invocation": sys.argv,
        "parsed_arguments": config_to_save,  # Include parsed args here as well
        "unitxt_version": unitxt_pkg_version,  # Use version from importlib.metadata
        "unitxt_commit_hash": unitxt_commit if unitxt_commit else "N/A",
        "python_version": platform.python_version(),
        "system": platform.system(),
        "system_version": platform.version(),
        "installed_packages": _get_installed_packages(),
    }

    # --- Prepare Final Results Structure ---
    results_summary = {
        "environment_info": environment_info,
        "results": global_scores,
    }

    # prepend to the results_path name the time in a wat like this: 2025-04-04T11:37:32

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    results_path = prepend_timestamp_to_path(results_path, timestamp)
    samples_path = prepend_timestamp_to_path(samples_path, timestamp)

    # --- Save Summary ---
    logger.info(f"Saving global results summary to: {results_path}")
    try:
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results_summary, f, indent=4, ensure_ascii=False)
    except OSError as e:
        logger.error(f"Failed to write results summary file {results_path}: {e}")
    except TypeError as e:
        logger.error(
            f"Failed to serialize results summary to JSON: {e}. Check data types."
        )
        # Log the problematic structure if possible (might be large)
        # logger.debug(f"Problematic results_summary structure: {results_summary}")

    # --- Save Samples (if requested) ---
    if args.log_samples:
        logger.info(f"Saving detailed samples to: {samples_path}")
        # Structure samples file with environment info as well for self-containment
        samples_output = {
            "environment_info": environment_info,  # Repeat env info here
            "samples": all_samples_data,
        }
        try:
            with open(samples_path, "w", encoding="utf-8") as f:
                json.dump(samples_output, f, indent=4, ensure_ascii=False)
        except OSError as e:
            logger.error(f"Failed to write samples file {samples_path}: {e}")
        except TypeError as e:
            logger.error(f"Failed to serialize samples to JSON: {e}. Check data types.")


def process_and_save_results(
    args: argparse.Namespace,
    evaluation_results: EvaluationResults,
    results_path: str,
    samples_path: str,
) -> None:
    """Processes, prints, and saves the evaluation results.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        evaluation_results (EvaluationResults): The list of evaluated instances.
        results_path (str): Path to save the summary results JSON file.
        samples_path (str): Path to save the detailed samples JSON file.

    Raises:
        Exception: If an error occurs during result processing or saving (re-raised).
    """
    try:
        # global_scores, all_samples_data = _extract_scores_and_samples(evaluated_dataset)

        subsets_scores = evaluation_results.subsets_scores
        instances_results = evaluation_results.instance_scores

        subset_instances = {}
        for instance in instances_results:
            if instance["subset"][0] not in subset_instances:
                subset_instances[instance["subset"][0]] = []
            del instance["postprocessors"]
            subset_instances[instance["subset"][0]].append(instance)

        logger.info(f"\n{subsets_scores.summary}")

        # --- Save Results ---
        # Pass all necessary data to the saving function
        _save_results_to_disk(
            args, subsets_scores, subset_instances, results_path, samples_path
        )

    except Exception:
        logger.exception(
            "An error occurred during result processing or saving"
        )  # Use logger.exception
        raise  # Re-raise after logging


def main():
    """Main function to parse arguments and run evaluation."""
    parser = setup_parser()
    args = parser.parse_args()

    # Setup logging ASAP
    setup_logging(args.verbosity)

    logger.info("Starting Unitxt Evaluation CLI")
    # Log raw and parsed args at DEBUG level
    logger.debug(f"Raw command line arguments: {sys.argv}")
    logger.debug(f"Parsed arguments: {vars(args)}")  # Log the vars(args) dict
    logger.debug(
        f"Parsed model_args type: {type(args.model_args)}, value: {args.model_args}"
    )

    try:
        results_path, samples_path = prepare_output_paths(
            args.output_path, args.output_file_prefix
        )

        # Apply unitxt settings within a context manager
        with configure_unitxt_settings(args):
            test_dataset = cli_load_dataset(args)
            model_args_dict = prepare_kwargs(args.model_args)
            gen_kwargs_dict = prepare_kwargs(args.gen_kwargs)
            chat_kwargs_dict = prepare_kwargs(args.chat_template_kwargs)

            model_args_dict.update(gen_kwargs_dict)
            inference_model = initialize_inference_engine(
                args, model_args_dict, chat_kwargs_dict
            )
            predictions = run_inference(inference_model, test_dataset)
            evaluation_results = run_evaluation(predictions, test_dataset)
            process_and_save_results(
                args, evaluation_results, results_path, samples_path
            )

    # --- More Specific Error Handling ---
    except (UnitxtArtifactNotFoundError, FileNotFoundError) as e:
        logger.exception(f"Error loading artifact or file: {e}")
        sys.exit(1)
    except (AttributeError, ValueError) as e:
        # Catch issues like missing keys in args, parsing errors, etc.
        logger.exception(f"Configuration or value error: {e}")
        sys.exit(1)
    except ImportError as e:
        # Catch missing optional dependencies
        logger.exception(f"Missing dependency: {e}")
        sys.exit(1)
    except RuntimeError as e:
        # Catch errors explicitly raised during execution (e.g., evaluation failure)
        logger.exception(f"Runtime error during processing: {e}")
        sys.exit(1)
    except Exception as e:
        # Catch any other unexpected errors
        logger.exception(f"An unexpected error occurred: {e}")
        sys.exit(1)

    logger.info("Unitxt Evaluation CLI finished successfully.")


def extract_scores(directory):  # pragma: no cover
    import pandas as pd

    data = []

    for filename in sorted(os.listdir(directory)):
        if filename.endswith("evaluation_results.json"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = json.load(f)

                    env_info = content.get("environment_info", {})
                    timestamp = env_info.get("timestamp_utc", "N/A")
                    model = env_info.get("parsed_arguments", {}).get("model", "N/A")
                    results = content.get("results", {})

                    row = {}
                    row["Model"] = model
                    row["Timestamp"] = timestamp
                    row["Average"] = results.get("score", "N/A")

                    for key in results.keys():
                        if isinstance(results[key], dict):
                            score = results[key].get("score", "N/A")
                            row[key] = score

                    data.append(row)
            except Exception as e:
                logger.error(f"Error parsing results file {filename}: {e}.")

    return pd.DataFrame(data).sort_values(by="Timestamp", ascending=True)


def summarize_cli():
    if len(sys.argv) != 2:
        logger.error("Usage: python summarize_cli_results.py <results-directory>")
        sys.exit(1)
    directory = sys.argv[1]
    df = extract_scores(directory)

    logger.info(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
