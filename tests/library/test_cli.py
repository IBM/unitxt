# test_evaluate_cli.py
import argparse
import json
import os
import unittest
from unittest.mock import (
    MagicMock,
    Mock,
    call,
    patch,
)

from datasets import Dataset as HFDataset
from unitxt import evaluate_cli as cli


class MockInferenceEngine:
    """A basic mock for an InferenceEngine."""

    def infer(self, dataset):
        # Return dummy predictions based on dataset length
        return [f"prediction_{i}" for i in range(len(dataset))]


class MockHFDataset:
    # Mock dataset class for testingclass MockHFDataset:
    """A basic mock for a Hugging Face Dataset."""

    def __init__(self, data=None):
        self._data = data if data else []

    def __len__(self):
        return len(self._data) if self._data else 5

    def __getitem__(self, idx):
        return self._data[idx]


class MockHFAutoModelInferenceEngine(MockInferenceEngine):
    """Mock for HF Engine."""

    pass


class MockCrossProviderInferenceEngine(MockInferenceEngine):
    """Mock for Cross Provider Engine."""

    pass


class TestUnitxtEvaluateCLI(unittest.TestCase):
    """Test suite for the evaluate_cli.py script."""

    def test_try_parse_json_valid(self):
        """Test try_parse_json with valid JSON and key-value strings."""
        self.assertEqual(
            cli.try_parse_json('{"key": "value", "num": 1}'),
            {"key": "value", "num": 1},
            "Should parse valid JSON.",
        )
        self.assertEqual(
            cli.try_parse_json("key=value,num=1"),
            {"key": "value", "num": 1},
            "Should parse valid key-value strings.",
        )
        self.assertEqual(
            cli.try_parse_json("just a string"),
            "just a string",
            "Should return string if not JSON or k=v.",
        )
        self.assertIsNone(
            cli.try_parse_json(None), "Should return None for None input."
        )

    def test_try_parse_json_invalid(self):
        """Test try_parse_json with invalid JSON."""
        with self.assertRaisesRegex(
            argparse.ArgumentTypeError,
            "Invalid JSON",
            msg="Should raise on invalid JSON syntax.",
        ):
            cli.try_parse_json('{key: "value"}')  # Missing quotes around key
        with self.assertRaisesRegex(
            argparse.ArgumentTypeError,
            "Invalid JSON",
            msg="Should raise on incomplete JSON.",
        ):
            cli.try_parse_json("[1, 2")  # Missing closing bracket

    def test_setup_parser(self):
        """Test if the argument parser is set up with expected arguments."""
        parser = cli.setup_parser()
        # Get actions by destination name for easier checking
        actions = {action.dest: action for action in parser._actions}

        # Check existence and properties of key arguments
        self.assertIn("tasks", actions)
        self.assertTrue(actions["tasks"].required, "--tasks should be required.")
        self.assertIn("model", actions)
        self.assertEqual(
            actions["model"].default, "hf", "--model default should be 'hf'."
        )
        self.assertIn("model_args", actions)
        self.assertEqual(
            actions["model_args"].type,
            cli.try_parse_json,
            "--model_args should use try_parse_json.",
        )
        self.assertIn("output_path", actions)
        self.assertEqual(
            actions["output_path"].default, ".", "--output_path default should be '.'."
        )
        self.assertIn("log_samples", actions)
        self.assertFalse(
            actions["log_samples"].default, "--log_samples default should be False."
        )
        self.assertIn("verbosity", actions)
        self.assertEqual(
            actions["verbosity"].default,
            "INFO",
            "--verbosity default should be 'INFO'.",
        )
        self.assertIn("trust_remote_code", actions)
        self.assertFalse(
            actions["trust_remote_code"].default,
            "--trust_remote_code default should be False.",
        )
        self.assertIn("num_fewshots", actions)  # Check for the new argument
        self.assertIsNone(
            actions["num_fewshots"].default, "--num_fewshots default should be None."
        )
        self.assertEqual(
            actions["num_fewshots"].type, int, "--num_fewshots should be int."
        )

    @patch("os.makedirs")
    def test_prepare_output_paths(self, mock_makedirs):
        """Test output path generation and directory creation."""
        results_path, samples_path = cli.prepare_output_paths(
            "/tmp/output", "my_results"
        )
        mock_makedirs.assert_called_once_with("/tmp/output", exist_ok=True)
        self.assertEqual(
            results_path,
            "/tmp/output/my_results.json",
            "Results path should be correctly formed.",
        )
        self.assertEqual(
            samples_path,
            "/tmp/output/my_results_samples.json",
            "Samples path should be correctly formed.",
        )

    @patch.object(cli, "settings")
    @patch("os.environ", {})  # Mock os.environ for isolation
    def test_configure_unitxt_settings(self, mock_settings):
        """Test configuration of unitxt settings."""
        args = argparse.Namespace(
            disable_hf_cache=True,
            trust_remote_code=True,
            cache_dir="/path/to/cache",
        )
        mock_context_manager = MagicMock()
        mock_settings.context.return_value = mock_context_manager

        with patch.object(cli, "logger"):  # Mock logger to avoid print during test
            context = cli.configure_unitxt_settings(args)

        self.assertEqual(
            context, mock_context_manager, "Should return the context manager."
        )
        mock_settings.context.assert_called_once_with(
            disable_hf_datasets_cache=True,
            allow_unverified_code=True,
            hf_cache_dir="/path/to/cache",
        )
        # Check environment variable setting
        self.assertEqual(
            os.environ.get("HF_DATASETS_CACHE"),
            "/path/to/cache",
            "Should set HF_DATASETS_CACHE environment variable.",
        )
        # Clean up environment variable after test
        if "HF_DATASETS_CACHE" in os.environ:
            del os.environ["HF_DATASETS_CACHE"]

    # --- Data Loading Tests ---
    # load_dataset is still patched in evaluate_cli as it's imported there at module level.
    @patch("unitxt.evaluate_cli.load_dataset")  # Patch load_dataset in evaluate_cli
    @patch("unitxt.evaluate_cli.Benchmark")  # Patch Benchmark in evaluate_cli
    @patch("unitxt.evaluate_cli.DatasetRecipe")
    @patch.object(cli, "logger")
    def test_load_data(
        self, mock_logger, mock_datasetdecipe, mock_benchmark, mock_load_dataset
    ):
        """Test data loading argument parsing and internal calls."""
        # --- Arrange ---
        # (Mock setup remains the same)
        mock_recipe_instance = MagicMock(name="MockRecipeInstance")
        mock_benchmark_instance = MagicMock(name="MockBenchmarkInstance")
        mock_final_dataset = MagicMock(spec=HFDataset, name="MockFinalDataset")

        mock_datasetdecipe.return_value = mock_recipe_instance
        mock_benchmark.return_value = mock_benchmark_instance
        mock_load_dataset.return_value = mock_final_dataset

        args = argparse.Namespace(
            tasks=["card=c,template=t"],
            split="validation",
            limit=5,
            num_fewshots=None,
            apply_chat_template=True,
        )

        expected_dataset_args = {
            "card": "c",
            "template": "t",
            "max_validation_instances": 5,
            "format": "formats.chat_api",
        }

        expected_benchmark_subsets = {"card=c,template=t": mock_recipe_instance}

        # --- Act ---
        # (Function call remains the same)
        returned_dataset = cli.cli_load_dataset(args)

        # --- Assert ---
        # (Assertions remain the same - they check the *logic* after import)
        mock_datasetdecipe.assert_called_once_with(**expected_dataset_args)
        mock_benchmark.assert_called_once_with(subsets=expected_benchmark_subsets)
        mock_load_dataset.assert_called_once_with(
            mock_benchmark_instance, split="validation"
        )
        self.assertEqual(
            returned_dataset,
            mock_final_dataset,
            "Should return the dataset from load_dataset.",
        )

    # --- Test: num_fewshots and limit arguments together ---
    @patch("unitxt.evaluate_cli.load_dataset")  # Patch load_dataset in evaluate_cli
    @patch("unitxt.evaluate_cli.Benchmark")  # Patch Benchmark in evaluate_cli
    @patch("unitxt.evaluate_cli.DatasetRecipe")
    @patch.object(cli, "logger")  # Keep logger mock
    def test_load_data_with_num_fewshots_and_limit(
        self, mock_logger, mock_datasetdecipe, mock_benchmark, mock_load_dataset
    ):
        """Test data loading with num_fewshots and limit arguments."""
        # --- Arrange ---
        mock_recipe_instance = MagicMock(name="MockRecipeInstance")
        mock_benchmark_instance = MagicMock(name="MockBenchmarkInstance")
        # Use spec=HFDataset if needed, ensure MockHFDataset is defined or use HFDataset
        mock_final_dataset = MagicMock(spec=HFDataset, name="MockFinalDataset")

        mock_datasetdecipe.return_value = mock_recipe_instance
        mock_benchmark.return_value = mock_benchmark_instance
        mock_load_dataset.return_value = mock_final_dataset

        args = argparse.Namespace(
            tasks=["card=c,template=t"],
            split="train",
            limit=10,  # Set limit
            num_fewshots=5,  # Set num_fewshots
            apply_chat_template=True,  # Assume True based on previous test
            # Add other necessary args if task_str_to_dataset_args uses them
        )

        # Calculate expected arguments for DatasetRecipe
        # based on task_str_to_dataset_args logic
        expected_dataset_args = {
            "card": "c",
            "template": "t",
            "max_train_instances": 10,  # From --limit and --split
            "num_demos": 5,  # From --num_fewshots
            "demos_taken_from": "train",  # Added when num_demos is set
            "demos_pool_size": -1,  # Added when num_demos is set
            "demos_removed_from_data": True,  # Added when num_demos is set
            "format": "formats.chat_api",  # From --apply_chat_template
        }

        expected_benchmark_subsets = {"card=c,template=t": mock_recipe_instance}

        # --- Act ---
        returned_dataset = cli.cli_load_dataset(args)

        # --- Assert ---
        # 1. Check DatasetRecipe call
        mock_datasetdecipe.assert_called_once_with(**expected_dataset_args)

        # 2. Check Benchmark call
        mock_benchmark.assert_called_once_with(subsets=expected_benchmark_subsets)

        # 3. Check load_dataset call
        mock_load_dataset.assert_called_once_with(
            mock_benchmark_instance,
            split="train",  # Use the split from args
        )

        # 4. Check final return value
        self.assertEqual(
            returned_dataset,
            mock_final_dataset,
            "Should return the dataset from load_dataset.",
        )

    # --- Test: prepare_kwargs function ---
    def test_prepare_kwargs(self):  # Renamed test function for clarity
        """Test model/gen/chat argument preparation (prepare_kwargs)."""
        # Test with dictionary input
        model_args_dict = {"key": "value", "num": 1}
        with patch.object(cli, "logger") as mock_logger_dict:
            # Pass the dictionary directly, as prepare_kwargs expects
            res = cli.prepare_kwargs(model_args_dict)
        self.assertEqual(
            res, {"key": "value", "num": 1}, "Should handle dict input directly."
        )
        # Check logger info call (optional, but good practice)
        mock_logger_dict.info.assert_called_with(f"Using kwargs: {model_args_dict}")

        # Test with parsed key-value string result (which is a dict)
        kv_parsed_dict = cli.try_parse_json("key=value,num=1,flag=true")
        with patch.object(cli, "logger") as mock_logger_kv:
            # Pass the resulting dictionary
            res_parsed = cli.prepare_kwargs(kv_parsed_dict)
        self.assertEqual(
            res_parsed,
            {"key": "value", "num": 1, "flag": True},
            "Should handle parsed k=v dict.",
        )
        mock_logger_kv.info.assert_called_with(f"Using kwargs: {kv_parsed_dict}")

        # Test with empty dict input
        empty_args_dict = {}
        with patch.object(cli, "logger") as mock_logger_empty:
            res_empty = cli.prepare_kwargs(empty_args_dict)
        self.assertEqual(res_empty, {}, "Should handle empty dict.")
        mock_logger_empty.info.assert_called_with("Using kwargs: {}")

        # Test with input that is not a dict (e.g., a string that try_parse_json didn't convert)
        # Note: try_parse_json handles kv strings, so this simulates passing a non-dict value directly
        unparsable_string = "this is not json or kv"
        with patch.object(cli, "logger") as mock_logger_unparsable:
            res_unparsable = cli.prepare_kwargs(unparsable_string)

        # According to prepare_kwargs logic, it should log a warning and return {}
        self.assertEqual(
            res_unparsable, {}, "Should return empty dict for non-dict input."
        )
        # Check the warning log
        mock_logger_unparsable.warning.assert_called_once()
        self.assertIn(
            f"Could not parse kwargs '{unparsable_string}' as JSON or key-value pairs. Treating as empty.",
            mock_logger_unparsable.warning.call_args[0][0],
        )
        # Check the info log (it should log the empty dict it's returning)
        mock_logger_unparsable.info.assert_called_with("Using kwargs: {}")

        # Test with None input
        with patch.object(cli, "logger") as mock_logger_none:
            res_none = cli.prepare_kwargs(None)
        self.assertEqual(res_none, {}, "Should return empty dict for None input.")
        # No warning expected for None
        mock_logger_none.warning.assert_not_called()
        mock_logger_none.info.assert_called_with("Using kwargs: {}")

    # --- Inference Engine Initialization Tests ---

    # Patch HFAutoModelInferenceEngine where it's looked up (evaluate_cli module)
    @patch("unitxt.evaluate_cli.HFAutoModelInferenceEngine")
    @patch.object(cli, "logger")  # Keep logger mock separate
    def test_initialize_inference_engine_hf_success(
        self, mock_logger, mock_hf_engine
    ):  # Correct order of args
        """Test initializing HF engine successfully."""
        # --- Arrange ---
        # Add batch_size to the args Namespace, mimicking argparse default
        args = argparse.Namespace(model="hf", batch_size=1)

        # Use copy() to avoid modification by the function if needed, good practice
        model_args_dict = {
            "pretrained": "model_id",
            "device": "cuda:0",
            "torch_dtype": "bfloat16",
        }
        # Define expected arguments passed to the engine constructor
        expected_engine_args = {
            "model_name": "model_id",
            "device": "cuda:0",
            "torch_dtype": "bfloat16",
            "batch_size": 1,  # <--- Expect batch_size to be added
            "chat_kwargs_dict": None,  # Expect chat_kwargs_dict from the call
        }

        # --- Act ---
        # Pass None or an empty dict for chat_kwargs_dict as in the original call
        engine = cli.initialize_inference_engine(
            args, model_args_dict.copy(), chat_kwargs_dict=None
        )

        # --- Assert ---
        # Check the engine type (optional, but confirms mock worked)
        self.assertIsInstance(
            engine, MagicMock, "Should return the mock engine instance."
        )

        # Check that HFAutoModelInferenceEngine was called correctly
        mock_hf_engine.assert_called_once_with(**expected_engine_args)

        # (Optional) Check specific logger calls if important
        # mock_logger.info.assert_any_call("Initializing HFAutoModelInferenceEngine for model: model_id")
        # mock_logger.info.assert_any_call(f"HFAutoModelInferenceEngine args: {expected_engine_args}")

    @patch("unitxt.evaluate_cli.HFAutoModelInferenceEngine")  # Patch where imported
    @patch.object(cli, "logger")  # Keep logger mock separate
    def test_initialize_inference_engine_hf_missing_pretrained(
        self, mock_logger, mock_hf_engine
    ):  # Args order corrected
        """Test initializing HF engine fails if 'pretrained' is missing."""
        # --- Arrange ---
        # Add batch_size for completeness, though not strictly needed before the expected error
        args = argparse.Namespace(model="hf", batch_size=1)
        model_args_dict = {"device": "cpu"}  # Missing 'pretrained'

        # --- Act & Assert ---
        with self.assertRaisesRegex(
            ValueError,
            "'pretrained' is required",  # Check if message matches exactly
            # Optional msg if assertion fails:
            # msg="Should raise ValueError if pretrained is missing.",
        ):
            # Provide the missing chat_kwargs_dict argument
            cli.initialize_inference_engine(
                args, model_args_dict.copy(), chat_kwargs_dict=None
            )

        # Assert the engine was not called because the error should happen first
        mock_hf_engine.assert_not_called()
        # (Optional) Assert logger error call
        # mock_logger.error.assert_called_once_with(...)

    # Correct patch target and added chat_kwargs_dict to call
    @patch("unitxt.evaluate_cli.CrossProviderInferenceEngine")  # Patch where imported
    @patch.object(cli, "logger")  # Keep logger mock separate
    def test_initialize_inference_engine_cross_provider_success(
        self, mock_logger, mock_cp_engine
    ):  # Args order corrected
        """Test initializing CrossProvider engine successfully."""
        # --- Arrange ---
        args = argparse.Namespace(model="cross_provider")  # batch_size not needed here
        model_args_dict = {
            "model_name": "provider/model",
            "max_tokens": 512,
            "temperature": 0.8,
        }
        # Define expected arguments for the engine constructor
        expected_engine_args = {
            "model": "provider/model",  # Note: passed as 'model', not 'model_name'
            "max_tokens": 512,
            "temperature": 0.8,
        }

        # --- Act ---
        # Provide the missing chat_kwargs_dict argument
        engine = cli.initialize_inference_engine(
            args,
            model_args_dict.copy(),
            chat_kwargs_dict=None,  # Added chat_kwargs_dict
        )

        # --- Assert ---
        self.assertIsInstance(
            engine, MagicMock, "Should return a mock engine instance."
        )
        # Check the engine constructor call
        mock_cp_engine.assert_called_once_with(**expected_engine_args)
        # (Optional) Check logger calls
        # mock_logger.info.assert_any_call(...)

    # Correct patch target and added chat_kwargs_dict to call
    @patch("unitxt.evaluate_cli.CrossProviderInferenceEngine")  # Patch where imported
    @patch.object(cli, "logger")  # Keep logger mock separate
    def test_initialize_inference_engine_cross_provider_missing_model_name(
        self,
        mock_logger,
        mock_cp_engine,  # Args order corrected
    ):
        """Test initializing CrossProvider engine fails if 'model_name' is missing."""
        # --- Arrange ---
        args = argparse.Namespace(model="cross_provider")  # batch_size not needed
        model_args_dict = {"max_tokens": 100}  # Missing 'model_name'

        # --- Act & Assert ---
        with self.assertRaisesRegex(
            ValueError,
            "'model_name' is required",  # Check if message matches exactly
            # Optional msg if assertion fails:
            # msg="Should raise ValueError if model_name is missing.",
        ):
            # Provide the missing chat_kwargs_dict argument
            cli.initialize_inference_engine(
                args,
                model_args_dict.copy(),
                chat_kwargs_dict=None,  # Added chat_kwargs_dict
            )

        # Assert the engine was not called
        mock_cp_engine.assert_not_called()
        # (Optional) Assert logger error call
        # mock_logger.error.assert_called_once_with(...)

    # --- Inference Execution ---
    # Patch the base class method if engines inherit from it, or mock the instance directly
    @patch.object(cli.InferenceEngine, "infer")
    def test_run_inference_success(self, mock_infer_method):
        """Test inference runner success."""
        # Create a mock instance *of the specific type* if needed, or use a generic MagicMock
        mock_engine_instance = MagicMock(spec=cli.InferenceEngine)
        # Configure the mock 'infer' method ON THE INSTANCE
        mock_engine_instance.infer.return_value = ["pred1", "pred2"]
        mock_dataset = MagicMock(spec=MockHFDataset)  # Use spec
        mock_dataset.__len__.return_value = 2  # Match prediction count

        with patch.object(cli, "logger"):
            predictions = cli.run_inference(mock_engine_instance, mock_dataset)

        self.assertEqual(predictions, ["pred1", "pred2"], "Should return predictions.")
        mock_engine_instance.infer.assert_called_once_with(mock_dataset)

    @patch.object(cli.InferenceEngine, "infer")
    def test_run_inference_mismatch_length(self, mock_infer_method):
        """Test inference runner logs error on prediction/dataset length mismatch."""
        mock_engine_instance = MagicMock(spec=cli.InferenceEngine)
        mock_engine_instance.infer.return_value = ["pred1"]  # Only one prediction
        mock_dataset = MagicMock(spec=MockHFDataset)
        mock_dataset.__len__.return_value = 2  # Dataset has two items

        with patch.object(cli, "logger") as mock_logger:
            predictions = cli.run_inference(mock_engine_instance, mock_dataset)

        self.assertEqual(predictions, ["pred1"])  # Should still return predictions
        mock_logger.error.assert_called_once()
        self.assertIn(
            "unexpected number of predictions", mock_logger.error.call_args[0][0]
        )

    @patch.object(cli.InferenceEngine, "infer")
    def test_run_inference_failure(self, mock_infer_method):
        """Test inference runner failure raises exception."""
        mock_engine_instance = MagicMock(spec=cli.InferenceEngine)
        mock_engine_instance.infer.side_effect = ConnectionError("API failed")
        mock_dataset = MagicMock(spec=MockHFDataset)
        mock_dataset.__len__.return_value = 1

        with self.assertRaises(ConnectionError, msg="Should re-raise the exception."):
            with patch.object(cli, "logger") as mock_logger:
                cli.run_inference(mock_engine_instance, mock_dataset)
                # Check exception was logged *before* re-raising
                mock_logger.exception.assert_called_once()
                self.assertIn(
                    "error occurred during inference",
                    mock_logger.exception.call_args[0][0],
                )

    # --- Evaluation Execution ---
    @patch("unitxt.evaluate_cli.evaluate")  # Correct patch target
    @patch.object(cli, "logger")  # Keep logger mock separate
    def test_run_evaluation_success(
        self, mock_logger, mock_evaluate
    ):  # Correct arg order
        """Test the evaluation runner successfully."""
        # --- Arrange ---
        mock_predictions = ["pred1", "pred2"]
        mock_dataset = MagicMock(spec=HFDataset)  # Use spec=HFDataset if appropriate

        # Create a mock OBJECT that simulates EvaluationResults
        mock_eval_results_obj = Mock(spec=cli.EvaluationResults)
        # Set attributes that might be checked or used later
        # (Based on process_and_save_results, these are needed)
        mock_eval_results_obj.subsets_scores = Mock(summary="Mock Scores")
        mock_eval_results_obj.instance_scores = [  # Keep instance data structure
            {"subset": ["task"], "score": 1.0, "postprocessors": []},
            {"subset": ["task"], "score": 0.0, "postprocessors": []},
        ]
        # Set the return value of the mocked evaluate function
        mock_evaluate.return_value = mock_eval_results_obj

        # --- Act ---
        results = cli.run_evaluation(mock_predictions, mock_dataset)

        # --- Assert ---
        # 1. Check that the mock EvaluationResults object was returned
        self.assertEqual(
            results,
            mock_eval_results_obj,
            "Should return the mock EvaluationResults object.",
        )
        # 2. Check that the underlying evaluate function was called correctly
        mock_evaluate.assert_called_once_with(
            predictions=mock_predictions, data=mock_dataset
        )
        # 3. Check logger calls (optional)
        mock_logger.info.assert_any_call("Starting evaluation...")
        mock_logger.info.assert_any_call("Evaluation completed.")
        # Ensure the error log for wrong type wasn't called
        mock_logger.error.assert_not_called()

    @patch.object(cli, "evaluate")
    def test_run_evaluation_no_predictions(self, mock_evaluate):
        """Test evaluation runner handles empty predictions."""
        mock_predictions = []
        mock_dataset = MagicMock(spec=MockHFDataset)

        with patch.object(cli, "logger") as mock_logger:
            results = cli.run_evaluation(mock_predictions, mock_dataset)

        self.assertEqual(results, [], "Should return empty list if no predictions.")
        mock_evaluate.assert_not_called()
        mock_logger.warning.assert_called_once()
        self.assertIn("Skipping evaluation", mock_logger.warning.call_args[0][0])

    @patch.object(cli, "evaluate")
    def test_run_evaluation_failure(self, mock_evaluate):
        """Test the evaluation runner when evaluate fails."""
        mock_predictions = ["pred1"]
        mock_dataset = MagicMock(spec=MockHFDataset)
        mock_evaluate.side_effect = ValueError("Evaluation failed")

        with self.assertRaises(ValueError, msg="Should re-raise evaluation exception."):
            with patch.object(cli, "logger") as mock_logger:
                cli.run_evaluation(mock_predictions, mock_dataset)
                # Check exception was logged *before* re-raising
                mock_logger.exception.assert_called_once()
                self.assertIn(
                    "error occurred during evaluation",
                    mock_logger.exception.call_args[0][0],
                )

    @patch.object(cli, "evaluate")
    def test_run_evaluation_returns_empty(self, mock_evaluate):
        """Test evaluation runner raises error if evaluate returns empty list/None."""
        mock_predictions = ["pred1"]
        mock_dataset = MagicMock(spec=MockHFDataset)
        mock_evaluate.return_value = []  # Simulate empty result

        with self.assertRaisesRegex(
            RuntimeError,
            "returned no results",
            msg="Should raise RuntimeError on empty results.",
        ):
            with patch.object(cli, "logger") as mock_logger:
                cli.run_evaluation(mock_predictions, mock_dataset)
                mock_logger.error.assert_called_once()  # Check error was logged

    @patch.object(cli, "evaluate")
    def test_run_evaluation_returns_wrong_type(self, mock_evaluate):
        """Test evaluation runner raises error if evaluate returns non-list."""
        mock_predictions = ["pred1"]
        mock_dataset = MagicMock(spec=MockHFDataset)
        mock_evaluate.return_value = {"wrong": "type"}  # Simulate wrong return type

        with self.assertRaisesRegex(
            RuntimeError,
            "unexpected type",
            msg="Should raise RuntimeError on wrong return type.",
        ):
            with patch.object(cli, "logger") as mock_logger:
                cli.run_evaluation(mock_predictions, mock_dataset)
                mock_logger.error.assert_called_once()  # Check error was logged

    # --- Result Processing/Saving Tests ---

    @patch("unitxt.evaluate_cli._save_results_to_disk")
    @patch.object(cli, "logger")
    def test_process_and_save_results_success(self, mock_logger, mock_save_results):
        """Test process_and_save_results with typical valid input."""
        # --- Arrange ---
        # Mock args (only need attributes used by _save_results_to_disk if any, e.g. log_samples)
        mock_args = argparse.Namespace(log_samples=True)  # Assume log_samples is used

        # Mock EvaluationResults object
        mock_eval_results = Mock(name="MockEvaluationResults")

        # Mock subset scores (needs a .summary attribute for logging)
        mock_subset_scores = Mock(name="MockSubsetScores")
        mock_subset_scores.summary = "Overall Score Summary"
        mock_eval_results.subsets_scores = mock_subset_scores

        # Mock instance scores
        mock_eval_results.instance_scores = [
            {
                "subset": ["task1"],
                "score": 1.0,
                "postprocessors": ["pp1"],
                "other_data": "a",
            },
            {
                "subset": ["task1"],
                "score": 0.5,
                "postprocessors": ["pp1"],
                "other_data": "b",
            },
            {
                "subset": ["task2"],
                "score": 0.8,
                "postprocessors": ["pp2"],
                "other_data": "c",
            },
        ]

        # Expected structure passed to _save_results_to_disk
        expected_subset_instances = {
            "task1": [
                {
                    "subset": ["task1"],
                    "score": 1.0,
                    "other_data": "a",
                },  # 'postprocessors' removed
                {
                    "subset": ["task1"],
                    "score": 0.5,
                    "other_data": "b",
                },  # 'postprocessors' removed
            ],
            "task2": [
                {
                    "subset": ["task2"],
                    "score": 0.8,
                    "other_data": "c",
                },  # 'postprocessors' removed
            ],
        }

        results_path = "/fake/path/results.json"
        samples_path = "/fake/path/samples.json"

        # --- Act ---
        cli.process_and_save_results(
            mock_args, mock_eval_results, results_path, samples_path
        )

        # --- Assert ---
        # 1. Check summary logging
        # Note: The code prepends a newline: logger.info(f"\n{subsets_scores.summary}")
        mock_logger.info.assert_called_once_with("\nOverall Score Summary")

        # 2. Check call to _save_results_to_disk
        mock_save_results.assert_called_once_with(
            mock_args,  # Pass args through
            mock_subset_scores,  # Pass subsets_scores through
            expected_subset_instances,  # Check the processed instance data
            results_path,  # Pass path through
            samples_path,  # Pass path through
        )

    @patch("unitxt.evaluate_cli._save_results_to_disk")
    @patch.object(cli, "logger")
    def test_process_and_save_results_empty_instances(
        self, mock_logger, mock_save_results
    ):
        """Test process_and_save_results when instance_scores list is empty."""
        # --- Arrange ---
        mock_args = argparse.Namespace(log_samples=False)
        mock_eval_results = Mock(name="MockEvaluationResults")
        mock_subset_scores = Mock(name="MockSubsetScores")
        mock_subset_scores.summary = "Summary With No Instances"
        mock_eval_results.subsets_scores = mock_subset_scores
        mock_eval_results.instance_scores = []  # Empty list

        results_path = "/fake/path/results2.json"
        samples_path = "/fake/path/samples2.json"

        # Expected structure passed to _save_results_to_disk
        expected_subset_instances = {}  # Should be an empty dict

        # --- Act ---
        cli.process_and_save_results(
            mock_args, mock_eval_results, results_path, samples_path
        )

        # --- Assert ---
        # 1. Check summary logging
        mock_logger.info.assert_called_once_with("\nSummary With No Instances")

        # 2. Check call to _save_results_to_disk
        mock_save_results.assert_called_once_with(
            mock_args,
            mock_subset_scores,
            expected_subset_instances,  # Expect empty dict here
            results_path,
            samples_path,
        )

    @patch("unitxt.evaluate_cli._save_results_to_disk")
    @patch.object(cli, "logger")
    def test_process_and_save_results_removes_postprocessors(
        self, mock_logger, mock_save_results
    ):
        """Test that 'postprocessors' key is removed from instances."""
        # --- Arrange ---
        mock_args = argparse.Namespace(log_samples=True)
        mock_eval_results = Mock(name="MockEvaluationResults")
        mock_subset_scores = Mock(name="MockSubsetScores")
        mock_subset_scores.summary = "Summary For Postprocessor Check"
        mock_eval_results.subsets_scores = mock_subset_scores
        # Instance score explicitly includes 'postprocessors'
        mock_eval_results.instance_scores = [
            {
                "subset": ["taskA"],
                "score": 1.0,
                "postprocessors": ["some_processor"],
                "data": "x",
            }
        ]

        results_path = "/fake/path/results3.json"
        samples_path = "/fake/path/samples3.json"

        # Expected structure passed to _save_results_to_disk (no 'postprocessors')
        expected_subset_instances = {
            "taskA": [{"subset": ["taskA"], "score": 1.0, "data": "x"}]
        }

        # --- Act ---
        cli.process_and_save_results(
            mock_args, mock_eval_results, results_path, samples_path
        )

        # --- Assert ---
        # 1. Check summary logging
        mock_logger.info.assert_called_once_with("\nSummary For Postprocessor Check")

        # 2. Check call to _save_results_to_disk, specifically the instance data
        mock_save_results.assert_called_once_with(
            mock_args,
            mock_subset_scores,
            expected_subset_instances,  # Verify the structure here
            results_path,
            samples_path,
        )
        # Optional: More detailed check on the structure passed if needed
        call_args, call_kwargs = mock_save_results.call_args
        passed_instances = call_args[2]  # Get the subset_instances dict
        self.assertIn("taskA", passed_instances)
        self.assertEqual(len(passed_instances["taskA"]), 1)
        self.assertNotIn(
            "postprocessors",
            passed_instances["taskA"][0],
            "'postprocessors' key should have been removed",
        )

    # In TestUnitxtEvaluateCLI class:
    # (Imports: datetime, json, sys, platform assumed present)

    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    @patch("unitxt.evaluate_cli.datetime")  # Patch datetime where it's used
    @patch("unitxt.evaluate_cli._get_unitxt_version", return_value="0.1.0")
    @patch("unitxt.evaluate_cli._get_unitxt_commit_hash", return_value="dummy_hash")
    @patch("unitxt.evaluate_cli._get_installed_packages", return_value={"pkg1": "1.0"})
    @patch.object(cli, "logger")
    def test_save_results_to_disk_summary_only(
        self,
        mock_logger,
        mock_get_pkgs,
        mock_get_hash,
        mock_get_ver,
        mock_datetime,
        mock_open,  # Corresponds to @patch("builtins.open"...)
    ):
        """Test saving only the summary results file (log_samples=False)."""
        # --- Arrange ---
        # (Arrange section remains the same as previous version)
        mock_timestamp = "2025-04-14T10:00:00"
        mock_now = MagicMock()
        mock_now.strftime.return_value = mock_timestamp
        mock_datetime.now.return_value = mock_now
        mock_utcnow = MagicMock()
        mock_utcnow.isoformat.return_value = "2025-04-14T08:00:00"
        mock_datetime.utcnow.return_value = mock_utcnow

        args = argparse.Namespace(
            log_samples=False,
            tasks=["card=x"],
            model="hf",
            model_args={"pretrained": "model"},
            output_path="/out",
            output_file_prefix="results_prefix",
            verbosity="INFO",
            split="test",
            limit=None,
            num_fewshots=None,
            batch_size=1,
            gen_kwargs=None,
            chat_template_kwargs=None,
            apply_chat_template=False,
            trust_remote_code=False,
            disable_hf_cache=False,
            cache_dir=None,
        )
        global_scores = {"accuracy": 0.8}
        subset_instances_data = {
            "task1": [{"metrics": {"acc": 1.0}, "other": "data1"}],
            "task2": [{"metrics": {"acc": 0.6}, "other": "data2"}],
        }
        base_results_path = "/out/results_prefix.json"
        base_samples_path = "/out/results_prefix_samples.json"
        expected_timestamped_results_path = f"/out/{mock_timestamp}_results_prefix.json"

        # --- Act ---
        cli._save_results_to_disk(
            args,
            global_scores,
            subset_instances_data,
            base_results_path,
            base_samples_path,
        )

        # --- Assert ---
        # 1. Assert summary file was opened with the timestamped name
        mock_open.assert_called_once_with(
            expected_timestamped_results_path, "w", encoding="utf-8"
        )

        # 2. Get the mock file handle instance *returned by mock_open()*
        mock_file_handle = mock_open()

        # 3. Reconstruct the full string written from ALL calls to write
        # Check if write was called at all (it should be if dump was called)
        self.assertTrue(
            mock_file_handle.write.called,
            "Expected file handle's write method to be called.",
        )
        # Get all arguments passed to write calls
        all_write_calls = mock_file_handle.write.call_args_list
        # Join the first argument (the string) from each call
        full_written_string = "".join(call[0][0] for call in all_write_calls)

        # 4. Parse the reconstructed JSON data
        try:
            saved_data = json.loads(full_written_string)
        except json.JSONDecodeError as e:
            self.fail(
                f"Failed to parse reconstructed JSON written to mock file: {e}\nData:\n{full_written_string}"
            )

        # 5. Check structure of the saved summary data (same checks as before)
        self.assertIn("environment_info", saved_data)
        self.assertIn("results", saved_data)
        self.assertEqual(saved_data["results"], global_scores)

        env_info = saved_data["environment_info"]
        self.assertIn("parsed_arguments", env_info)
        self.assertEqual(env_info["parsed_arguments"]["tasks"], ["card=x"])
        self.assertEqual(env_info["unitxt_version"], "0.1.0")
        self.assertEqual(env_info["unitxt_commit_hash"], "dummy_hash")
        self.assertEqual(env_info["installed_packages"], {"pkg1": "1.0"})
        self.assertEqual(env_info["timestamp_utc"], "2025-04-14T08:00:00Z")
        self.assertIn("python_version", env_info)
        self.assertIn("system", env_info)

        # 6. Assert logger calls (same checks as before)
        mock_logger.info.assert_any_call(
            f"Saving global results summary to: {expected_timestamped_results_path}"
        )
        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        expected_timestamped_samples_path = (
            f"/out/{mock_timestamp}_results_prefix_samples.json"
        )
        self.assertNotIn(
            f"Saving detailed samples to: {expected_timestamped_samples_path}",
            log_calls,
        )

    # --- Tests for main ---

    @patch("unitxt.evaluate_cli.setup_parser")
    @patch("unitxt.evaluate_cli.setup_logging")
    @patch("unitxt.evaluate_cli.prepare_output_paths")
    @patch("unitxt.evaluate_cli.configure_unitxt_settings")
    @patch("unitxt.evaluate_cli.cli_load_dataset")  # Patched correct function
    @patch("unitxt.evaluate_cli.prepare_kwargs")  # Patched correct function
    @patch("unitxt.evaluate_cli.initialize_inference_engine")
    @patch("unitxt.evaluate_cli.run_inference")
    @patch("unitxt.evaluate_cli.run_evaluation")
    @patch("unitxt.evaluate_cli.process_and_save_results")
    @patch("sys.exit")
    @patch.object(cli, "logger")  # Patch logger last
    def test_main_success_flow(
        self,
        mock_logger,  # Corresponds to @patch.object(cli, "logger")
        mock_exit,  # Corresponds to @patch("sys.exit")
        mock_process_save,  # Corresponds to @patch.object(cli, "process_and_save_results")
        mock_run_eval,  # Corresponds to @patch.object(cli, "run_evaluation")
        mock_run_infer,  # Corresponds to @patch.object(cli, "run_inference")
        mock_init_engine,  # Corresponds to @patch.object(cli, "initialize_inference_engine")
        mock_prepare_kwargs,  # Corresponds to @patch('unitxt.evaluate_cli.prepare_kwargs')
        mock_cli_load_dataset,  # Corresponds to @patch('unitxt.evaluate_cli.cli_load_dataset')
        mock_configure_settings,  # Corresponds to @patch.object(cli, "configure_unitxt_settings")
        mock_prep_paths,  # Corresponds to @patch.object(cli, "prepare_output_paths")
        mock_setup_logging,  # Corresponds to @patch.object(cli, "setup_logging")
        mock_setup_parser,  # Corresponds to @patch.object(cli, "setup_parser")
    ):
        """Test the main function success path."""
        # --- Arrange Mocks ---

        # 1. Mock parser and args
        mock_parser = MagicMock(spec=argparse.ArgumentParser)
        # Provide a full Namespace reflecting a typical run
        mock_args = argparse.Namespace(
            verbosity="INFO",
            output_path="mock_output",
            output_file_prefix="mock_results",
            tasks=[
                "card=dummy,template=dummy"
            ],  # Note: tasks is list now due to type=partial(...)
            split="test",
            limit=None,
            num_fewshots=None,
            batch_size=1,
            model="hf",
            model_args={
                "pretrained": "mock-hf-model",
                "device": "cpu",
            },  # Dict from try_parse_json
            gen_kwargs={"temperature": 0.0},  # Dict from try_parse_json
            chat_template_kwargs=None,  # Example None
            log_samples=True,
            trust_remote_code=False,
            disable_hf_cache=False,
            cache_dir=None,
            apply_chat_template=False,
        )
        mock_parser.parse_args.return_value = mock_args
        mock_setup_parser.return_value = mock_parser

        # 2. Mock path prep
        mock_results_path = "./mock_output/mock_results.json"
        mock_samples_path = "./mock_output/mock_results_samples.json"
        mock_prep_paths.return_value = (mock_results_path, mock_samples_path)

        # 3. Mock settings context manager
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = None
        mock_context_manager.__exit__.return_value = None  # Indicate no error on exit
        mock_configure_settings.return_value = mock_context_manager

        # 4. Mock dataset loading (cli_load_dataset)
        mock_dataset_instance = Mock(spec=HFDataset)
        mock_dataset_instance.__len__ = MagicMock(return_value=1)
        mock_cli_load_dataset.return_value = mock_dataset_instance

        # 5. Mock prepare_kwargs (it's called 3 times)
        # We'll have it return the dict passed in args, or {} for None
        def prepare_kwargs_side_effect(kwargs_input):
            if isinstance(kwargs_input, dict):
                return kwargs_input
            if kwargs_input is None:
                return {}
            # Should not happen with valid args setup
            return {}

        mock_prepare_kwargs.side_effect = prepare_kwargs_side_effect
        # Expected results from prepare_kwargs calls in main
        expected_model_args_prep = {"pretrained": "mock-hf-model", "device": "cpu"}
        expected_gen_kwargs_prep = {"temperature": 0.0}
        expected_chat_kwargs_prep = {}

        # 6. Mock engine initialization
        mock_engine_instance = MagicMock(
            spec=cli.InferenceEngine
        )  # Use spec if possible
        mock_init_engine.return_value = mock_engine_instance
        # Expected combined dict for engine init (model_args + gen_kwargs)
        expected_engine_model_args = expected_model_args_prep.copy()
        expected_engine_model_args.update(expected_gen_kwargs_prep)

        # 7. Mock inference
        mock_predictions = ["pred1"]
        mock_run_infer.return_value = mock_predictions

        # 8. Mock evaluation (returns EvaluationResults object)
        mock_eval_results_obj = Mock(spec=cli.EvaluationResults)
        # Set attributes needed by process_and_save_results
        mock_eval_results_obj.subsets_scores = Mock(summary="Mock Summary")
        mock_eval_results_obj.instance_scores = [
            {"subset": ["task"], "postprocessors": []}
        ]  # Example structure
        mock_run_eval.return_value = mock_eval_results_obj

        # --- Act ---
        cli.main()

        # --- Assert ---
        mock_setup_parser.assert_called_once()
        mock_parser.parse_args.assert_called_once()
        mock_setup_logging.assert_called_once_with("INFO")
        mock_prep_paths.assert_called_once_with("mock_output", "mock_results")
        mock_configure_settings.assert_called_once_with(mock_args)
        mock_context_manager.__enter__.assert_called_once()

        # Assert cli_load_dataset called
        mock_cli_load_dataset.assert_called_once_with(mock_args)

        # Assert prepare_kwargs called 3 times correctly
        self.assertEqual(mock_prepare_kwargs.call_count, 3)
        mock_prepare_kwargs.assert_has_calls(
            [
                call(mock_args.model_args),
                call(mock_args.gen_kwargs),
                call(mock_args.chat_template_kwargs),
            ],
            any_order=False,
        )  # Order matters here

        # Assert initialize_inference_engine called correctly
        mock_init_engine.assert_called_once_with(
            mock_args,  # Pass args through
            expected_engine_model_args,  # Pass combined model+gen args
            expected_chat_kwargs_prep,  # Pass prepared chat args
        )

        mock_run_infer.assert_called_once_with(
            mock_engine_instance, mock_dataset_instance
        )
        mock_run_eval.assert_called_once_with(mock_predictions, mock_dataset_instance)

        # Assert process_and_save_results called correctly
        mock_process_save.assert_called_once_with(
            mock_args,
            mock_eval_results_obj,  # Pass the EvaluationResults object
            mock_results_path,
            mock_samples_path,
        )

        mock_context_manager.__exit__.assert_called_once()
        mock_exit.assert_not_called()  # No exit on success

        mock_logger.info.assert_any_call("Starting Unitxt Evaluation CLI")
        mock_logger.info.assert_any_call("Unitxt Evaluation CLI finished successfully.")

    @patch("unitxt.evaluate_cli.setup_parser")
    @patch("unitxt.evaluate_cli.setup_logging")
    @patch("unitxt.evaluate_cli.prepare_output_paths")
    @patch("unitxt.evaluate_cli.configure_unitxt_settings")
    @patch("unitxt.evaluate_cli.cli_load_dataset")
    @patch("unitxt.evaluate_cli.prepare_kwargs")
    @patch("unitxt.evaluate_cli.initialize_inference_engine")
    @patch("unitxt.evaluate_cli.run_inference")
    @patch("unitxt.evaluate_cli.run_evaluation")
    @patch("unitxt.evaluate_cli.process_and_save_results")
    @patch("sys.exit")
    @patch.object(cli, "logger")  # Patch logger last
    def test_main_bird_remote_scenario(
        self,
        mock_logger,  # logger
        mock_exit,  # sys.exit
        mock_process_save,  # process_and_save_results
        mock_run_eval,  # run_evaluation
        mock_run_infer,  # run_inference
        mock_init_engine,  # initialize_inference_engine
        mock_prepare_kwargs,  # prepare_kwargs
        mock_cli_load_dataset,  # cli_load_dataset
        mock_configure_settings,  # configure_unitxt_settings
        mock_prep_paths,  # prepare_output_paths
        mock_setup_logging,  # setup_logging
        mock_setup_parser,  # setup_parser
    ):
        """Test the main function simulating the BIRD remote debug config."""
        # --- Arrange Mocks ---

        # 1. Mock parser setup and args for BIRD scenario
        mock_parser = MagicMock(spec=argparse.ArgumentParser)
        # model_args as would be returned by try_parse_json
        parsed_model_args_input = {
            "model_name": "llama-3-3-70b-instruct",
            "max_tokens": 256,
        }
        mock_args = argparse.Namespace(
            # tasks is now a list based on parser setup
            tasks=[
                "card=cards.text2sql.bird,template=templates.text2sql.you_are_given_with_hint_with_sql_prefix"
            ],
            model="cross_provider",
            model_args=parsed_model_args_input,  # Already a dict
            split="validation",
            limit=100,
            num_fewshots=None,
            batch_size=1,  # Add batch_size, even if not used by cross_provider path
            gen_kwargs=None,
            chat_template_kwargs=None,
            output_path="./debug_output/bird_remote",
            output_file_prefix="evaluation_results",
            log_samples=True,
            verbosity="INFO",
            trust_remote_code=True,
            disable_hf_cache=False,
            cache_dir=None,
            apply_chat_template=False,
        )
        mock_parser.parse_args.return_value = mock_args
        mock_setup_parser.return_value = mock_parser

        # 2. Mock path preparation
        expected_results_path = "./debug_output/bird_remote/evaluation_results.json"
        expected_samples_path = (
            "./debug_output/bird_remote/evaluation_results_samples.json"
        )
        mock_prep_paths.return_value = (expected_results_path, expected_samples_path)

        # 3. Mock settings context manager
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = None
        mock_context_manager.__exit__.return_value = None
        mock_configure_settings.return_value = mock_context_manager

        # 4. Mock data loading (cli_load_dataset)
        mock_dataset_instance = Mock(spec=HFDataset)
        # Correctly mock __len__
        mock_dataset_instance.__len__ = MagicMock(return_value=2)
        mock_cli_load_dataset.return_value = mock_dataset_instance

        # 5. Mock prepare_kwargs calls
        def prepare_kwargs_side_effect(kwargs_input):
            if kwargs_input == parsed_model_args_input:
                return parsed_model_args_input  # Return the dict itself
            if kwargs_input is None:
                return {}  # Return empty dict for None inputs
            return {}

        mock_prepare_kwargs.side_effect = prepare_kwargs_side_effect
        # Expected results from prepare_kwargs calls
        expected_model_args_prep = parsed_model_args_input
        expected_gen_kwargs_prep = {}
        expected_chat_kwargs_prep = {}

        # 6. Mock engine initialization
        mock_remote_engine_instance = MagicMock(spec=cli.InferenceEngine)
        mock_init_engine.return_value = mock_remote_engine_instance
        # Expected combined dict for engine init (model_args + gen_kwargs)
        expected_engine_model_args = expected_model_args_prep.copy()
        expected_engine_model_args.update(
            expected_gen_kwargs_prep
        )  # No change here as gen_kwargs is {}

        # 7. Mock inference
        mock_predictions = ["sql pred 1", "sql pred 2"]
        mock_run_infer.return_value = mock_predictions

        # 8. Mock evaluation (returns EvaluationResults object)
        mock_eval_results_obj = Mock(spec=cli.EvaluationResults)
        mock_eval_results_obj.subsets_scores = Mock(summary="BIRD Summary")
        # Provide instance scores matching the expected format
        mock_eval_results_obj.instance_scores = [
            {
                "subset": ["task_bird"],
                "score": 0.0,
                "postprocessors": [],
                "source": "q0",
                "prediction": "sql pred 1",
            },
            {
                "subset": ["task_bird"],
                "score": 1.0,
                "postprocessors": [],
                "source": "q1",
                "prediction": "sql pred 2",
            },
        ]
        mock_run_eval.return_value = mock_eval_results_obj

        # --- Act ---
        cli.main()

        # --- Assert ---
        mock_setup_parser.assert_called_once()
        mock_parser.parse_args.assert_called_once()
        mock_setup_logging.assert_called_once_with("INFO")
        mock_prep_paths.assert_called_once_with(
            "./debug_output/bird_remote", "evaluation_results"
        )
        mock_configure_settings.assert_called_once_with(mock_args)
        self.assertTrue(
            mock_configure_settings.call_args[0][0].trust_remote_code
        )  # Check specific arg
        mock_context_manager.__enter__.assert_called_once()

        # Check cli_load_dataset call
        mock_cli_load_dataset.assert_called_once_with(mock_args)
        # You could add more specific checks here if needed, e.g., that limit=100 was passed
        # self.assertEqual(mock_cli_load_dataset.call_args[0][0].limit, 100)

        # Check prepare_kwargs calls
        self.assertEqual(mock_prepare_kwargs.call_count, 3)
        mock_prepare_kwargs.assert_has_calls(
            [
                call(mock_args.model_args),
                call(mock_args.gen_kwargs),
                call(mock_args.chat_template_kwargs),
            ],
            any_order=False,
        )

        # Check initialize_inference_engine call
        mock_init_engine.assert_called_once_with(
            mock_args,  # The Namespace object
            expected_engine_model_args,  # model_args dict merged with gen_kwargs dict
            expected_chat_kwargs_prep,  # chat_kwargs dict
        )
        # Check specific args passed if needed
        self.assertEqual(mock_init_engine.call_args[0][0].model, "cross_provider")

        mock_run_infer.assert_called_once_with(
            mock_remote_engine_instance, mock_dataset_instance
        )
        mock_run_eval.assert_called_once_with(mock_predictions, mock_dataset_instance)

        # Check process_and_save_results call
        mock_process_save.assert_called_once_with(
            mock_args,
            mock_eval_results_obj,  # Pass the EvaluationResults object
            expected_results_path,
            expected_samples_path,
        )
        self.assertTrue(
            mock_process_save.call_args[0][0].log_samples
        )  # Check specific arg

        mock_context_manager.__exit__.assert_called_once()
        mock_exit.assert_not_called()  # No exit on success
        mock_logger.info.assert_any_call("Starting Unitxt Evaluation CLI")
        mock_logger.info.assert_any_call("Unitxt Evaluation CLI finished successfully.")

    # --- Test for main function: Local HF Model Scenario ---
    @patch("unitxt.evaluate_cli.setup_parser")
    @patch("unitxt.evaluate_cli.setup_logging")
    @patch("unitxt.evaluate_cli.prepare_output_paths")
    @patch("unitxt.evaluate_cli.configure_unitxt_settings")
    @patch("unitxt.evaluate_cli.cli_load_dataset")
    @patch("unitxt.evaluate_cli.prepare_kwargs")
    @patch("unitxt.evaluate_cli.initialize_inference_engine")
    @patch("unitxt.evaluate_cli.run_inference")
    @patch("unitxt.evaluate_cli.run_evaluation")
    @patch("unitxt.evaluate_cli.process_and_save_results")
    @patch("sys.exit")
    @patch.object(cli, "logger")  # Patch logger last
    def test_main_local_hf_scenario(
        self,
        mock_logger,  # logger
        mock_exit,  # sys.exit
        mock_process_save,  # process_and_save_results
        mock_run_eval,  # run_evaluation
        mock_run_infer,  # run_inference
        mock_init_engine,  # initialize_inference_engine
        mock_prepare_kwargs,  # prepare_kwargs
        mock_cli_load_dataset,  # cli_load_dataset
        mock_configure_settings,  # configure_unitxt_settings
        mock_prep_paths,  # prepare_output_paths
        mock_setup_logging,  # setup_logging
        mock_setup_parser,  # setup_parser
    ):
        """Test the main function success path simulating a local HF model."""
        # --- Arrange Mocks ---

        # 1. Mock parser setup and args for local HF scenario
        mock_parser = MagicMock(spec=argparse.ArgumentParser)
        # model_args dict as if parsed from JSON/kv string
        parsed_model_args_input = {
            "pretrained": "gpt2",  # Required for model='hf'
            "device": "cpu",
            "torch_dtype": "float32",  # Example args
        }
        # gen_kwargs dict as if parsed
        parsed_gen_kwargs_input = {"max_length": 50}
        # chat_kwargs dict as if parsed
        parsed_chat_kwargs_input = {"add_generation_prompt": True}

        mock_args = argparse.Namespace(
            tasks=[
                "card=cards.sst2,template=templates.classification.multi_choice.standard"
            ],  # Example task
            model="hf",  # Local model type
            model_args=parsed_model_args_input,
            split="test",
            limit=50,
            num_fewshots=None,
            batch_size=4,  # Example batch size for HF
            gen_kwargs=parsed_gen_kwargs_input,  # Include some gen kwargs
            chat_template_kwargs=parsed_chat_kwargs_input,  # Include some chat kwargs
            output_path="./output/local_hf",
            output_file_prefix="hf_sst2_results",
            log_samples=False,  # Example: disable sample logging
            verbosity="DEBUG",  # Example: use different verbosity
            trust_remote_code=False,
            disable_hf_cache=False,
            cache_dir=None,
            apply_chat_template=False,
        )
        mock_parser.parse_args.return_value = mock_args
        mock_setup_parser.return_value = mock_parser

        # 2. Mock path preparation
        expected_results_path = "./output/local_hf/hf_sst2_results.json"
        expected_samples_path = "./output/local_hf/hf_sst2_results_samples.json"
        mock_prep_paths.return_value = (expected_results_path, expected_samples_path)

        # 3. Mock settings context manager
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = None
        mock_context_manager.__exit__.return_value = None
        mock_configure_settings.return_value = mock_context_manager

        # 4. Mock data loading (cli_load_dataset)
        mock_dataset_instance = Mock(spec=HFDataset)
        mock_dataset_instance.__len__ = MagicMock(return_value=50)  # Match limit
        mock_cli_load_dataset.return_value = mock_dataset_instance

        # 5. Mock prepare_kwargs calls
        def prepare_kwargs_side_effect(kwargs_input):
            # Return the dict directly if it's the input, otherwise {} for None
            if kwargs_input == parsed_model_args_input:
                return parsed_model_args_input
            if kwargs_input == parsed_gen_kwargs_input:
                return parsed_gen_kwargs_input
            if kwargs_input == parsed_chat_kwargs_input:
                return parsed_chat_kwargs_input
            if kwargs_input is None:
                return {}
            return {}  # Default fallback

        mock_prepare_kwargs.side_effect = prepare_kwargs_side_effect
        # Expected results from prepare_kwargs calls
        expected_model_args_prep = parsed_model_args_input
        expected_gen_kwargs_prep = parsed_gen_kwargs_input
        expected_chat_kwargs_prep = parsed_chat_kwargs_input

        # 6. Mock engine initialization
        mock_local_engine_instance = MagicMock(spec=cli.InferenceEngine)
        mock_init_engine.return_value = mock_local_engine_instance
        # Expected combined dict for engine init (model_args + gen_kwargs)
        expected_engine_model_args = expected_model_args_prep.copy()
        expected_engine_model_args.update(expected_gen_kwargs_prep)

        # 7. Mock inference
        mock_predictions = [f"pred_{i}" for i in range(50)]  # Match dataset length
        mock_run_infer.return_value = mock_predictions

        # 8. Mock evaluation (returns EvaluationResults object)
        mock_eval_results_obj = Mock(spec=cli.EvaluationResults)
        mock_eval_results_obj.subsets_scores = Mock(summary="Local HF Summary")
        mock_eval_results_obj.instance_scores = [  # Example structure
            {
                "subset": ["sst2"],
                "score": 1.0,
                "postprocessors": [],
                "prediction": f"pred_{i}",
            }
            for i in range(50)
        ]
        mock_run_eval.return_value = mock_eval_results_obj

        # --- Act ---
        cli.main()

        # --- Assert ---
        mock_setup_parser.assert_called_once()
        mock_parser.parse_args.assert_called_once()
        mock_setup_logging.assert_called_once_with(
            "DEBUG"
        )  # Check correct verbosity used
        mock_prep_paths.assert_called_once_with("./output/local_hf", "hf_sst2_results")
        mock_configure_settings.assert_called_once_with(mock_args)
        mock_context_manager.__enter__.assert_called_once()
        mock_cli_load_dataset.assert_called_once_with(mock_args)
        # self.assertEqual(mock_cli_load_dataset.call_args[0][0].limit, 50) # Optional check

        # Check prepare_kwargs calls
        self.assertEqual(mock_prepare_kwargs.call_count, 3)
        mock_prepare_kwargs.assert_has_calls(
            [
                call(mock_args.model_args),
                call(mock_args.gen_kwargs),
                call(mock_args.chat_template_kwargs),
            ],
            any_order=False,
        )

        # Check initialize_inference_engine call
        mock_init_engine.assert_called_once_with(
            mock_args,  # The Namespace object
            expected_engine_model_args,  # model_args dict merged with gen_kwargs dict
            expected_chat_kwargs_prep,  # chat_kwargs dict
        )
        # Check specific args passed if needed
        self.assertEqual(mock_init_engine.call_args[0][0].model, "hf")
        self.assertEqual(mock_init_engine.call_args[0][1]["pretrained"], "gpt2")

        mock_run_infer.assert_called_once_with(
            mock_local_engine_instance, mock_dataset_instance
        )
        mock_run_eval.assert_called_once_with(mock_predictions, mock_dataset_instance)
        mock_process_save.assert_called_once_with(
            mock_args,
            mock_eval_results_obj,  # Pass the EvaluationResults object
            expected_results_path,
            expected_samples_path,
        )
        # Check specific args passed if needed
        self.assertFalse(
            mock_process_save.call_args[0][0].log_samples
        )  # Check log_samples

        mock_context_manager.__exit__.assert_called_once()
        mock_exit.assert_not_called()  # No exit on success
        mock_logger.info.assert_any_call("Starting Unitxt Evaluation CLI")
        mock_logger.info.assert_any_call("Unitxt Evaluation CLI finished successfully.")


if __name__ == "__main__":
    unittest.main()
