# unitxt_evaluate_cli.py
import argparse
import importlib.util  # Added for checking optional dependencies
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

from datasets import Dataset as HFDataset  # Added for type hinting

# Use relative imports assuming the script is inside the 'unitxt' package
from . import evaluate, get_logger, load_dataset

# Import UnitxtArtifactNotFoundError from artifact instead of error_utils
from .artifact import UnitxtArtifactNotFoundError

# Use HFAutoModelInferenceEngine for local models
from .inference import (
    CrossProviderInferenceEngine,
    HFAutoModelInferenceEngine,
    InferenceEngine,  # Added for type hinting
)

# Corrected import for settings
from .settings_utils import settings
from .text_utils import print_dict

# Define logger early so it can be used in initial error handling
# Basic config for initial messages, will be reconfigured in main()
logger = get_logger()


def _package_is_available(package_name: str) -> bool:
    """Checks if a package is available without importing it."""
    return importlib.util.find_spec(package_name) is not None


def _parse_key_value_string(value: str) -> Optional[Dict[str, Any]]:
    """Parses a key=value comma-separated string into a dict."""
    parsed_dict = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split("=", 1)
        if len(parts) == 2:
            key, val_str = parts
            key = key.strip()  # Strip whitespace from key
            # Attempt to convert value to int, float, or bool
            val: Any
            try:
                val = int(val_str)
            except ValueError:
                try:
                    val = float(val_str)
                except ValueError:
                    if val_str.lower() == "true":
                        val = True
                    elif val_str.lower() == "false":
                        val = False
                    else:
                        val = val_str  # Keep as string
            parsed_dict[key] = val
        else:
            logger.warning(
                f"Could not parse argument part: '{item}'. Expected format 'key=value'."
            )
    return parsed_dict if parsed_dict else None


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
            parsed_dict = _parse_key_value_string(value)
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
        # Fixed B904: Added 'from e'
        raise argparse.ArgumentTypeError(f"Could not parse argument: '{value}'") from e


def setup_parser() -> argparse.ArgumentParser:
    """Sets up the argument parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="CLI utility for running evaluations with unitxt.",
    )

    # --- Task/Dataset Arguments ---
    parser.add_argument(
        "--tasks",
        "-t",
        type=str,
        required=True,
        help="Unitxt task/dataset identifier string.\n"
        "Format: 'card=<card_ref>,template=<template_ref>,...'\n"
        "Example: 'card=cards.mmlu,template=templates.mmlu.all_5_shot'",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (e.g., 'train', 'validation', 'test'). Default: 'test'.",
    )
    parser.add_argument(
        "--limit",
        "-L",
        type=int,
        default=None,
        metavar="N",
        help="Limit the number of examples per task/dataset.",
    )

    # --- Model Arguments (Explicit Types) ---
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="hf",
        choices=["hf", "generic_remote"],
        help="Specifies the model type/engine.\n"
        "- 'hf': Local Hugging Face model via HFAutoModel (default). Requires 'pretrained=...' in --model_args.\n"
        "- 'generic_remote': Remote model via CrossProviderInferenceEngine. Requires 'model_name=...' in --model_args.",
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

    # --- Output and Logging Arguments ---
    parser.add_argument(
        "--output_dir",
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
        force=True,
    )
    logger.setLevel(verbosity)


def prepare_output_paths(output_dir: str, prefix: str) -> Tuple[str, str]:
    """Creates output directory and defines file paths.

    Args:
        output_dir (str): The directory where output files will be saved.
        prefix (str): The prefix for the output file names.

    Returns:
        Tuple[str, str]: A tuple containing the path for the results summary file
                         and the path for the detailed samples file.
    """
    os.makedirs(output_dir, exist_ok=True)
    results_file_path = os.path.join(output_dir, f"{prefix}.json")
    samples_file_path = os.path.join(output_dir, f"{prefix}_samples.json")
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
        os.environ["HF_DATASETS_CACHE"] = args.cache_dir
    logger.info(f"Applying unitxt settings: {unitxt_settings_dict}")
    return settings.context(**unitxt_settings_dict)


def load_data(args: argparse.Namespace) -> HFDataset:
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
    dataset_args_str = args.tasks
    if args.limit is not None:
        if "limit=" not in dataset_args_str and "loader_limit=" not in dataset_args_str:
            dataset_args_str += f",loader_limit={args.limit}"  # Use loader_limit for unitxt compatibility
        else:
            logger.warning(
                f"Limit specified in both --tasks string and --limit arg. Using --limit={args.limit}."
            )
            dataset_args_str = re.sub(
                r"(loader_)?limit=\d+",
                f"loader_limit={args.limit}",  # Use loader_limit for unitxt compatibility
                dataset_args_str,
            )

    test_dataset = load_dataset(dataset_args_str, split=args.split)
    logger.info(
        f"Dataset loaded successfully. Number of instances: {len(test_dataset)}"
    )
    return test_dataset


def prepare_model_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Prepares the model arguments dictionary.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Dict[str, Any]: The processed model arguments dictionary.
    """
    model_args_dict = args.model_args if isinstance(args.model_args, dict) else {}
    logger.info(f"Using model_args: {model_args_dict}")
    return model_args_dict


def initialize_inference_engine(
    args: argparse.Namespace, model_args_dict: Dict[str, Any]
) -> InferenceEngine:
    """Initializes the appropriate inference engine based on arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        model_args_dict (Dict[str, Any]): Processed model arguments.

    Returns:
        InferenceEngine: The initialized inference engine instance.

    Raises:
        SystemExit: If required dependencies are missing for the selected model type.
        ValueError: If required keys are missing in model_args for the selected model type.
    """
    inference_model = None
    # --- Local Hugging Face Model (using HFAutoModelInferenceEngine) ---
    if args.model.lower() == "hf":
        # Check for transformers and torch only when hf model is selected
        if (
            not _package_is_available("transformers")
            or not _package_is_available("torch")
            or not _package_is_available("accelerate")
        ):
            logger.error(
                "Packages 'transformers' 'accelerate' and 'torch' are required for '--model hf'."
                " Please install them (`pip install transformers torch accelerate`)."
            )
            sys.exit(1)

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

        device_arg = model_args_dict.pop("device", None)

        inference_model = HFAutoModelInferenceEngine(
            model_name=local_model_name, device=device_arg, **model_args_dict
        )

    # --- Remote Model (CrossProviderInferenceEngine) ---
    elif args.model.lower() == "generic_remote":
        # Check for litellm and tenacity only when generic_remote model is selected
        if not _package_is_available("litellm"):
            logger.error(
                "Package 'litellm' is required for '--model generic_remote'."
                " Please install it (`pip install litellm`)."
            )
            sys.exit(1)
        if not _package_is_available("tenacity"):
            logger.error(
                "Package 'tenacity' is required for '--model generic_remote'."
                " Please install it (`pip install tenacity`)."
            )
            sys.exit(1)

        if "model_name" not in model_args_dict:
            logger.error(
                "Missing 'model_name=<provider/model_id>' in --model_args for '--model generic_remote'."
            )
            raise ValueError(
                "Argument 'model_name' is required in --model_args when --model is 'generic_remote'"
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

        # Note: CrossProviderInferenceEngine expects 'model' parameter, not 'model_name'
        inference_model = CrossProviderInferenceEngine(
            model=remote_model_name,
            **model_args_dict,
        )
    else:
        # This case should not be reached due to argparse choices
        logger.error(
            f"Invalid --model type specified: {args.model}. Use 'hf' or 'generic_remote'."
        )
        sys.exit(1)

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
        if not predictions or len(predictions) != len(dataset):
            logger.error(
                f"Inference returned an unexpected number of predictions ({len(predictions)}). Expected {len(dataset)}."
            )
            # Potentially exit: sys.exit(1) # Consider if this should be fatal
        return predictions
    except Exception:
        logger.exception("An error occurred during inference")  # Use logger.exception
        raise  # Re-raise after logging


def run_evaluation(predictions: List[Any], dataset: HFDataset) -> List[Dict[str, Any]]:
    """Runs evaluation on the predictions.

    Args:
        predictions (List[Any]): The list of predictions from the model.
        dataset (HFDataset): The dataset containing references and other data.

    Returns:
        List[Dict[str, Any]]: The evaluated dataset (list of instances with scores).

    Raises:
        Exception: If an error occurs during evaluation or if evaluation returns no results.
    """
    logger.info("Starting evaluation...")
    try:
        evaluated_dataset = evaluate(predictions=predictions, data=dataset)
        logger.info("Evaluation completed.")
        if not evaluated_dataset:
            logger.error("Evaluation returned no results.")
            raise RuntimeError(
                "Evaluation returned no results."
            )  # Raise instead of sys.exit
        return evaluated_dataset
    except Exception:
        logger.exception("An error occurred during evaluation")  # Use logger.exception
        raise  # Re-raise after logging


def _extract_scores_and_samples(
    evaluated_dataset: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Extracts global scores and sample data from the evaluated dataset.

    Args:
        evaluated_dataset (List[Dict[str, Any]]): The list of evaluated instances.

    Returns:
        Tuple[Dict[str, Any], List[Dict[str, Any]]]: A tuple containing the global scores
                                                     and the list of processed sample data.
    """
    global_scores = {}
    all_samples_data = []

    if isinstance(evaluated_dataset, list) and evaluated_dataset:
        if (
            "score" in evaluated_dataset[0]
            and "global" in evaluated_dataset[0]["score"]
        ):
            global_scores = evaluated_dataset[0]["score"]["global"]
            logger.info("Global scores found in the first evaluated instance.")
        else:
            logger.warning(
                "Could not automatically locate global scores in evaluated_dataset[0]['score']['global']."
            )

        for i, instance in enumerate(evaluated_dataset):
            instance_result = {
                "index": i,
                "source": instance.get("source"),
                "prediction": instance.get("prediction"),
                "references": instance.get("references"),
                "metrics": instance.get("score", {}).get("instance", {}),
                "task_data": instance.get("task_data"),
            }
            all_samples_data.append(instance_result)

    elif isinstance(evaluated_dataset, dict):
        logger.warning(
            "Evaluation returned a dictionary, structure might differ. Attempting to find scores."
        )
        global_scores = evaluated_dataset.get("global_scores", {})
    else:
        logger.error(f"Evaluation returned unexpected type: {type(evaluated_dataset)}")
        raise TypeError(
            f"Evaluation returned unexpected type: {type(evaluated_dataset)}"
        )

    return global_scores, all_samples_data


def _save_results_to_disk(
    args: argparse.Namespace,
    global_scores: Dict[str, Any],
    all_samples_data: List[Dict[str, Any]],
    results_path: str,
    samples_path: str,
) -> None:
    """Saves the configuration, global scores, and samples to JSON files.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        global_scores (Dict[str, Any]): Dictionary of global scores.
        all_samples_data (List[Dict[str, Any]]): List of processed sample data.
        results_path (str): Path to save the summary results JSON file.
        samples_path (str): Path to save the detailed samples JSON file.
    """
    config_to_save = {}
    for k, v in vars(args).items():
        if isinstance(v, (str, int, float, bool, list, dict, type(None))):
            config_to_save[k] = v
        else:
            config_to_save[k] = repr(v)  # Represent other types as string

    results_summary = {
        "config": config_to_save,
        "global_scores": global_scores,
    }

    logger.info(f"Saving global results summary to: {results_path}")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=4, ensure_ascii=False)

    if args.log_samples:
        logger.info(f"Saving detailed samples to: {samples_path}")
        with open(samples_path, "w", encoding="utf-8") as f:
            json.dump(all_samples_data, f, indent=4, ensure_ascii=False)


def process_and_save_results(
    args: argparse.Namespace,
    evaluated_dataset: List[Dict[str, Any]],
    results_path: str,
    samples_path: str,
) -> None:
    """Processes, prints, and saves the evaluation results.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        evaluated_dataset (List[Dict[str, Any]]): The list of evaluated instances.
        results_path (str): Path to save the summary results JSON file.
        samples_path (str): Path to save the detailed samples JSON file.

    Raises:
        Exception: If an error occurs during result processing or saving.
    """
    try:
        global_scores, all_samples_data = _extract_scores_and_samples(evaluated_dataset)

        logger.info("\n--- Global Scores ---")
        if global_scores:
            print_dict(global_scores)
        else:
            logger.info("No global scores found.")

        logger.info("\n--- Example Instance (Index 0) ---")
        if all_samples_data:
            print_dict(
                all_samples_data[0],
                keys_to_print=["source", "prediction", "references", "metrics"],
            )
        else:
            logger.info("No instances found in the evaluated dataset.")

        # --- Save Results ---
        _save_results_to_disk(
            args, global_scores, all_samples_data, results_path, samples_path
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

    setup_logging(args.verbosity)
    logger.info("Starting Unitxt Evaluation CLI")
    logger.debug(f"Raw arguments: {sys.argv}")
    logger.debug(f"Parsed arguments: {args}")
    logger.debug(
        f"Parsed model_args type: {type(args.model_args)}, value: {args.model_args}"
    )

    results_path, samples_path = prepare_output_paths(
        args.output_dir, args.output_file_prefix
    )

    with configure_unitxt_settings(args):
        try:
            test_dataset = load_data(args)
            model_args_dict = prepare_model_args(args)
            inference_model = initialize_inference_engine(args, model_args_dict)
            predictions = run_inference(inference_model, test_dataset)
            evaluated_dataset = run_evaluation(predictions, test_dataset)
            process_and_save_results(
                args, evaluated_dataset, results_path, samples_path
            )
        except (
            UnitxtArtifactNotFoundError,
            FileNotFoundError,
            AttributeError,
            ValueError,
        ) as e:
            # Log specific, potentially recoverable errors
            logger.exception(f"Error during setup or data processing: {e}")
            sys.exit(1)
        except ImportError as e:
            # Log missing dependency errors
            logger.exception(f"Missing dependency: {e}")
            sys.exit(1)
        except RuntimeError as e:
            # Log runtime errors during evaluation/inference
            logger.exception(f"Runtime error during processing: {e}")
            sys.exit(1)
        except Exception as e:
            # Log unexpected errors
            logger.exception(f"An unexpected error occurred: {e}")
            sys.exit(1)

    logger.info("Unitxt Evaluation CLI finished.")


if __name__ == "__main__":
    main()
