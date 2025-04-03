# test_evaluate_cli.py
import argparse
import os
import unittest
from unittest.mock import (
    MagicMock,
    call,
    patch,
)

# Assuming evaluate_cli is in the same directory or package
# If evaluate_cli.py is in the parent directory:
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from unitxt import evaluate_cli as cli
# If evaluate_cli is part of an installed package:
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

    # --- Parsing and Setup Tests ---
    def test_parse_key_value_string(self):
        """Test the _parse_key_value_string helper function."""
        self.assertEqual(
            cli._parse_key_value_string(
                "key1=value1,key2=123,key3=4.5,key4=True,key5=false"
            ),
            {"key1": "value1", "key2": 123, "key3": 4.5, "key4": True, "key5": False},
            "Should parse various data types correctly.",
        )
        self.assertIsNone(
            cli._parse_key_value_string(""), "Should return None for empty string."
        )
        self.assertIsNone(
            cli._parse_key_value_string("key_no_value"),
            "Should return None for key without value.",
        )
        # Check logging on invalid part (use assertLogs)
        with self.assertLogs(cli.logger, level="WARNING") as cm:
            result = cli._parse_key_value_string("key=val1,invalid,key2=val2")
        self.assertEqual(
            result,
            {"key": "val1", "key2": "val2"},
            "Should parse valid parts even if one part is invalid.",
        )
        self.assertIn(
            "Could not parse argument part: 'invalid'", cm.output[0]
        )  # Check warning log

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
        self.assertIn("task", actions)
        self.assertTrue(actions["task"].required, "--task should be required.")
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
    @patch.object(cli, "load_dataset")
    def test_load_data(self, mock_load_dataset):
        """Test data loading function with limit."""
        mock_dataset = MagicMock(spec=MockHFDataset)  # Use spec for better mocking
        mock_load_dataset.return_value = mock_dataset

        args = argparse.Namespace(
            task="card=c,template=t",
            split="validation",
            limit=5,
            num_fewshots=None,  # Explicitly None
            apply_chat_template=True,
        )
        with patch.object(cli, "logger"):  # Mock logger
            dataset = cli.load_data(args)

        self.assertEqual(dataset, mock_dataset, "Should return the loaded dataset.")
        mock_load_dataset.assert_called_once_with(
            "card=c,template=t,loader_limit=5,format=formats.chat_api",
            split="validation",
        )

    # --- NEW TEST: Test num_fewshots argument ---
    @patch.object(cli, "load_dataset")
    def test_load_data_with_num_fewshots(self, mock_load_dataset):
        """Test data loading function with num_fewshots argument."""
        mock_dataset = MagicMock(spec=MockHFDataset)
        mock_load_dataset.return_value = mock_dataset

        args = argparse.Namespace(
            task="card=c,template=t",
            split="test",
            limit=None,  # No limit
            num_fewshots=3,  # Set num_fewshots
            apply_chat_template=True,
        )
        with patch.object(cli, "logger"):  # Mock logger
            dataset = cli.load_data(args)

        self.assertEqual(dataset, mock_dataset, "Should return the loaded dataset.")
        # Check that num_demos is appended correctly
        mock_load_dataset.assert_called_once_with(
            "card=c,template=t,num_demos=3,format=formats.chat_api",
            split="test",
        )

    # --- NEW TEST: Test num_fewshots and limit arguments together ---
    @patch.object(cli, "load_dataset")
    def test_load_data_with_num_fewshots_and_limit(self, mock_load_dataset):
        """Test data loading function with both num_fewshots and limit arguments."""
        mock_dataset = MagicMock(spec=MockHFDataset)
        mock_load_dataset.return_value = mock_dataset

        args = argparse.Namespace(
            task="card=c,template=t",
            split="train",
            limit=10,  # Set limit
            num_fewshots=5,  # Set num_fewshots,
            apply_chat_template=True,
        )
        with patch.object(cli, "logger"):  # Mock logger
            dataset = cli.load_data(args)

        self.assertEqual(dataset, mock_dataset, "Should return the loaded dataset.")
        # Check that both loader_limit and num_demos are appended correctly
        mock_load_dataset.assert_called_once_with(
            "card=c,template=t,loader_limit=10,num_demos=5,format=formats.chat_api",
            split="train",
        )

    # --- Model Arg Prep ---
    def test_prepare_model_args(self):
        """Test model argument preparation."""
        args_dict = argparse.Namespace(model_args={"key": "value", "num": 1})
        with patch.object(cli, "logger"):  # Mock logger
            res = cli.prepare_model_args(args_dict)
        self.assertEqual(res, {"key": "value", "num": 1}, "Should handle dict input.")

        # Test with parsed key-value string
        args_kv_parsed = argparse.Namespace(
            model_args=cli.try_parse_json("key=value,num=1")
        )
        with patch.object(cli, "logger"):
            res_parsed = cli.prepare_model_args(args_kv_parsed)
        self.assertEqual(
            res_parsed, {"key": "value", "num": 1}, "Should handle parsed k=v."
        )

        args_empty = argparse.Namespace(model_args={})
        with patch.object(cli, "logger"):
            res_empty = cli.prepare_model_args(args_empty)
        self.assertEqual(res_empty, {}, "Should handle empty dict.")

        # Test with unparsable string (should log warning and return empty dict)
        args_unparsable = argparse.Namespace(model_args="this is not json or kv")
        with patch.object(cli, "logger") as mock_logger:
            res_unparsable = cli.prepare_model_args(args_unparsable)
        self.assertEqual(
            res_unparsable, {}, "Should return empty dict for unparsable string."
        )
        mock_logger.warning.assert_called_once()
        self.assertIn(
            "Could not parse --model_args", mock_logger.warning.call_args[0][0]
        )

    # --- Inference Engine Initialization Tests ---

    @patch.object(cli, "HFAutoModelInferenceEngine")
    def test_initialize_inference_engine_hf_success(self, mock_hf_engine):
        """Test initializing HF engine successfully."""
        args = argparse.Namespace(model="hf")
        # Use copy() to avoid modification by the function
        model_args_dict = {
            "pretrained": "model_id",
            "device": "cuda:0",
            "torch_dtype": "bfloat16",
        }
        with patch.object(cli, "logger"):
            engine = cli.initialize_inference_engine(args, model_args_dict.copy())

        self.assertIsInstance(
            engine, MagicMock, "Should return a mock engine instance."
        )
        # Check that 'pretrained' was removed and passed correctly, others passed as kwargs
        mock_hf_engine.assert_called_once_with(
            model_name="model_id", device="cuda:0", torch_dtype="bfloat16"
        )

    @patch.object(cli, "HFAutoModelInferenceEngine")
    def test_initialize_inference_engine_hf_missing_pretrained(self, mock_hf_engine):
        """Test initializing HF engine fails if 'pretrained' is missing."""
        args = argparse.Namespace(model="hf")
        model_args_dict = {"device": "cpu"}  # Missing 'pretrained'
        with self.assertRaisesRegex(
            ValueError,
            "'pretrained' is required",
            msg="Should raise ValueError if pretrained is missing.",
        ):
            with patch.object(cli, "logger"):
                cli.initialize_inference_engine(args, model_args_dict.copy())
        mock_hf_engine.assert_not_called()

    @patch.object(cli, "CrossProviderInferenceEngine")
    def test_initialize_inference_engine_cross_provider_success(self, mock_cp_engine):
        """Test initializing CrossProvider engine successfully."""
        args = argparse.Namespace(model="cross_provider")
        model_args_dict = {
            "model_name": "provider/model",
            "max_tokens": 512,
            "temperature": 0.8,
        }
        with patch.object(cli, "logger"):
            engine = cli.initialize_inference_engine(args, model_args_dict.copy())

        self.assertIsInstance(
            engine, MagicMock, "Should return a mock engine instance."
        )
        # Check 'model_name' was removed and passed as 'model', others as kwargs
        mock_cp_engine.assert_called_once_with(
            model="provider/model", max_tokens=512, temperature=0.8
        )

    @patch.object(cli, "CrossProviderInferenceEngine")
    def test_initialize_inference_engine_cross_provider_missing_model_name(
        self, mock_cp_engine
    ):
        """Test initializing CrossProvider engine fails if 'model_name' is missing."""
        args = argparse.Namespace(model="cross_provider")
        model_args_dict = {"max_tokens": 100}  # Missing 'model_name'
        with self.assertRaisesRegex(
            ValueError,
            "'model_name' is required",
            msg="Should raise ValueError if model_name is missing.",
        ):
            with patch.object(cli, "logger"):
                cli.initialize_inference_engine(args, model_args_dict.copy())
        mock_cp_engine.assert_not_called()

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
    @patch.object(cli, "evaluate")
    def test_run_evaluation_success(self, mock_evaluate):
        """Test the evaluation runner successfully."""
        mock_predictions = ["pred1", "pred2"]
        mock_dataset = MagicMock(spec=MockHFDataset)  # Use spec
        mock_eval_results = [
            {"score": {"global": {"acc": 1.0}, "instance": {"acc": 1.0}}},
            {"score": {"global": {"acc": 1.0}, "instance": {"acc": 1.0}}},
        ]
        mock_evaluate.return_value = mock_eval_results

        with patch.object(cli, "logger"):
            results = cli.run_evaluation(mock_predictions, mock_dataset)

        self.assertEqual(
            results, mock_eval_results, "Should return evaluation results."
        )
        mock_evaluate.assert_called_once_with(
            predictions=mock_predictions, data=mock_dataset
        )

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

    @patch.object(cli, "logger")
    def test_extract_scores_and_samples_success(self, mock_logger):
        """Test extracting scores and samples from evaluated data."""
        evaluated_data = [
            {
                "source": "s1",
                "prediction": "p1",
                "references": ["r1"],
                "score": {"global": {"acc": 1.0, "f1": 0.9}, "instance": {"acc": 1.0}},
                "task_data": {"id": 1},
                "other_key": "value",  # Should be ignored in sample output unless added
            },
            {
                "source": "s2",
                "prediction": "p2",
                "references": ["r2a", "r2b"],
                "score": {"global": {"acc": 1.0, "f1": 0.9}, "instance": {"acc": 0.0}},
                "task_data": {"id": 2},
            },
        ]
        global_scores, samples = cli._extract_scores_and_samples(evaluated_data)

        self.assertEqual(
            global_scores, {"acc": 1.0, "f1": 0.9}, "Should extract global scores."
        )
        self.assertEqual(len(samples), 2, "Should extract correct number of samples.")
        # Check structure of the first sample
        expected_sample_0 = {
            "index": 0,
            "source": "s1",
            "prediction": "p1",
            "references": ["r1"],
            "metrics": {"acc": 1.0},
            "task_data": {"id": 1},
        }
        self.assertEqual(
            samples[0], expected_sample_0, "First sample structure mismatch."
        )
        # Check structure of the second sample
        expected_sample_1 = {
            "index": 1,
            "source": "s2",
            "prediction": "p2",
            "references": ["r2a", "r2b"],
            "metrics": {"acc": 0.0},
            "task_data": {"id": 2},
        }
        self.assertEqual(
            samples[1], expected_sample_1, "Second sample structure mismatch."
        )

        mock_logger.warning.assert_not_called()  # No warnings expected

    @patch.object(cli, "logger")
    def test_extract_scores_and_samples_no_global(self, mock_logger):
        """Test extraction when global scores key is missing."""
        evaluated_data = [{"score": {"instance": {"acc": 1.0}}}]  # No 'global' key
        global_scores, samples = cli._extract_scores_and_samples(evaluated_data)

        self.assertEqual(global_scores, {}, "Global scores should be empty.")
        self.assertEqual(len(samples), 1, "Should still extract samples.")
        self.assertEqual(samples[0]["metrics"], {"acc": 1.0})
        mock_logger.warning.assert_called_once()
        self.assertIn(
            "Could not automatically locate global scores",
            mock_logger.warning.call_args[0][0],
            "Should log warning about missing global scores.",
        )

    @patch.object(cli, "logger")
    def test_extract_scores_and_samples_empty_global(self, mock_logger):
        """Test extraction when global scores dict is present but empty."""
        evaluated_data = [
            {"score": {"global": {}, "instance": {"acc": 1.0}}}
        ]  # Empty global
        global_scores, samples = cli._extract_scores_and_samples(evaluated_data)

        self.assertEqual(global_scores, {}, "Global scores should be empty.")
        self.assertEqual(len(samples), 1, "Should still extract samples.")
        mock_logger.warning.assert_called_once()
        self.assertIn(
            "Found 'score.global' key in first instance, but it was empty",
            mock_logger.warning.call_args[0][0],
            "Should log warning about empty global scores dict.",
        )

    @patch.object(cli, "logger")
    def test_extract_scores_and_samples_empty_input(self, mock_logger):
        """Test extraction with empty evaluated dataset."""
        evaluated_data = []
        global_scores, samples = cli._extract_scores_and_samples(evaluated_data)
        self.assertEqual(global_scores, {})
        self.assertEqual(samples, [])
        mock_logger.warning.assert_called_once_with(
            "Evaluated dataset is empty. No scores or samples to extract."
        )

    # Use unittest.mock.mock_open for file operations
    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    @patch.object(cli, "_get_unitxt_version", return_value="0.1.0")
    @patch.object(cli, "_get_unitxt_commit_hash", return_value="bla")
    @patch.object(cli, "_get_installed_packages", return_value={"pkg1": "1.0"})
    @patch.object(cli, "json")  # Mock json module
    @patch.object(cli, "logger")
    def test_save_results_to_disk_summary_only(
        self,
        mock_logger,
        mock_json,
        mock_get_pkgs,
        mock_get_hash,
        mock_get_ver,
        mock_open,
    ):
        """Test saving only the summary results file."""
        args = MagicMock(
            spec=argparse.Namespace, log_samples=False
        )  # log_samples is False
        # Set specific attributes needed for saving config
        args.task = "card=x"
        args.model = "hf"
        args.model_args = {"pretrained": "model"}
        args.output_path = "/out"
        # ... add other relevant args if they are saved

        global_scores = {"accuracy": 0.8}
        all_samples_data = [
            {"index": 0, "metrics": {}}
        ]  # Sample data is still processed
        results_path = "/out/results.json"
        samples_path = "/out/samples.json"

        cli._save_results_to_disk(
            args, global_scores, all_samples_data, results_path, samples_path
        )

        # Assert summary file was opened and written to
        mock_open.assert_called_once_with(results_path, "w", encoding="utf-8")
        # Check the structure passed to json.dump for the summary file
        self.assertEqual(mock_json.dump.call_count, 1)
        dump_args, dump_kwargs = mock_json.dump.call_args
        saved_data = dump_args[0]  # The object being dumped
        file_handle = dump_args[1]  # The file handle

        self.assertEqual(file_handle, mock_open())  # Check it's the correct handle
        self.assertIn("environment_info", saved_data)
        self.assertIn("global_scores", saved_data)
        self.assertEqual(saved_data["global_scores"], global_scores)
        self.assertIn("parsed_arguments", saved_data["environment_info"])
        self.assertEqual(
            saved_data["environment_info"]["parsed_arguments"]["task"], "card=x"
        )
        self.assertEqual(saved_data["environment_info"]["unitxt_version"], "0.1.0")
        self.assertEqual(saved_data["environment_info"]["unitxt_commit_hash"], "bla")
        self.assertEqual(
            saved_data["environment_info"]["installed_packages"], {"pkg1": "1.0"}
        )

        # Assert logger calls
        mock_logger.info.assert_any_call(
            f"Saving global results summary to: {results_path}"
        )
        # Ensure samples file logging did NOT happen
        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        self.assertNotIn(f"Saving detailed samples to: {samples_path}", log_calls)

    @patch.object(cli, "print_dict")
    @patch.object(cli, "_extract_scores_and_samples")
    @patch.object(cli, "_save_results_to_disk")
    @patch.object(cli, "logger")
    def test_process_and_save_results(
        self, mock_logger, mock_save, mock_extract, mock_print_dict
    ):
        """Test the main result processing and saving function."""
        # Arrange
        args = MagicMock(spec=argparse.Namespace, log_samples=True)
        evaluated_dataset = [{"score": {"global": {"acc": 1.0}, "instance": {}}}]
        results_path, samples_path = "res.json", "samp.json"
        mock_global_scores = {"acc": 1.0}
        mock_samples_data = [
            {
                "index": 0,
                "source": "s",
                "prediction": "p",
                "references": ["r"],
                "metrics": {},
            }
        ]
        mock_extract.return_value = (mock_global_scores, mock_samples_data)

        # Act
        cli.process_and_save_results(
            args, evaluated_dataset, results_path, samples_path
        )

        # Assert
        mock_extract.assert_called_once_with(evaluated_dataset)
        # Check print_dict calls
        self.assertEqual(mock_print_dict.call_count, 2)
        mock_print_dict.assert_has_calls(
            [
                call(mock_global_scores),  # Print global scores
                call(
                    mock_samples_data[0],
                    keys_to_print=["source", "prediction", "references", "metrics"],
                ),  # Print first sample
            ]
        )
        # Check save call
        mock_save.assert_called_once_with(
            args, mock_global_scores, mock_samples_data, results_path, samples_path
        )
        # Check logging info messages
        mock_logger.info.assert_has_calls(
            [
                call("\n--- Global Scores ---"),
                call("\n--- Example Instance (Index 0) ---"),
            ]
        )

    # --- Tests for main ---

    @patch.object(cli, "setup_parser")
    @patch.object(cli, "setup_logging")
    @patch.object(cli, "prepare_output_paths")
    @patch.object(cli, "configure_unitxt_settings")
    @patch.object(cli, "load_data")
    @patch.object(cli, "prepare_model_args")
    @patch.object(cli, "prepare_gen_kwargs")
    @patch.object(cli, "initialize_inference_engine")
    @patch.object(cli, "run_inference")
    @patch.object(cli, "run_evaluation")
    @patch.object(cli, "process_and_save_results")
    @patch("sys.exit")  # Keep patching sys.exit
    @patch.object(cli, "logger")  # Patch logger within cli module
    def test_main_success_flow(
        self,
        mock_logger,  # Corresponds to @patch.object(cli, "logger")
        mock_exit,  # Corresponds to @patch("sys.exit")
        mock_process_save,  # Corresponds to @patch.object(cli, "process_and_save_results")
        mock_run_eval,  # Corresponds to @patch.object(cli, "run_evaluation")
        mock_run_infer,  # Corresponds to @patch.object(cli, "run_inference")
        mock_init_engine,  # Corresponds to @patch.object(cli, "initialize_inference_engine")
        mock_prep_gen_kwargs,  # <<< Added mock argument
        mock_prep_model_args,  # Corresponds to @patch.object(cli, "prepare_model_args")
        mock_load_data,  # Corresponds to @patch.object(cli, "load_data")
        mock_configure_settings,  # Corresponds to @patch.object(cli, "configure_unitxt_settings")
        mock_prep_paths,  # Corresponds to @patch.object(cli, "prepare_output_paths")
        mock_setup_logging,  # Corresponds to @patch.object(cli, "setup_logging")
        mock_setup_parser,  # Corresponds to @patch.object(cli, "setup_parser")
    ):
        """Test the main function success path."""
        # --- Arrange Mocks ---

        # 1. Mock parser setup and argument parsing
        mock_parser = MagicMock(spec=argparse.ArgumentParser)
        # Define mock arguments that represent a valid success scenario
        mock_args = argparse.Namespace(
            verbosity="INFO",
            output_path="mock_output",  # Use a distinct name
            output_file_prefix="mock_results",
            task="card=dummy_card,template=dummy_template",  # Example task string
            split="test",
            limit=None,
            num_fewshots=None,
            model="hf",  # Using 'hf' model type
            # Provide required 'pretrained' key for 'hf' model
            model_args={"pretrained": "mock-hf-model", "device": "cpu"},
            gen_kwargs=None,  # Set explicitly for clarity
            log_samples=True,  # Test saving samples path
            trust_remote_code=False,
            disable_hf_cache=False,
            cache_dir=None,
            apply_chat_template=False,  # Include all defined args
        )
        mock_parser.parse_args.return_value = mock_args
        mock_setup_parser.return_value = mock_parser

        # 2. Mock path preparation
        mock_results_path = "./mock_output/mock_results.json"
        mock_samples_path = "./mock_output/mock_results_samples.json"
        mock_prep_paths.return_value = (mock_results_path, mock_samples_path)

        # 3. Mock unitxt settings context manager
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = None
        mock_context_manager.__exit__.return_value = None
        mock_configure_settings.return_value = mock_context_manager

        # 4. Mock data loading
        # Create a mock dataset with some structure if needed by later steps
        mock_dataset_instance = MockHFDataset([{"source": "q1", "references": ["a1"]}])
        mock_load_data.return_value = mock_dataset_instance

        # 5. Mock model and generation argument preparation
        # prepare_model_args returns the dict *passed* to the engine constructor
        # (after potentially popping keys like 'pretrained', 'device')
        mock_model_args_prepared = {"device": "cpu"}  # Example remaining args
        mock_prep_model_args.return_value = mock_model_args_prepared

        # Mock prepare_gen_kwargs (returns {} if args.gen_kwargs is None)
        mock_gen_kwargs_prepared = {}
        mock_prep_gen_kwargs.return_value = (
            mock_gen_kwargs_prepared  # <<< Set return value
        )

        # Calculate the final dict passed to initialize_inference_engine
        # This reflects the model_args_dict.update(gen_kwargs_dict) step in main
        combined_engine_args = mock_model_args_prepared.copy()
        combined_engine_args.update(
            mock_gen_kwargs_prepared
        )  # Still {"device": "cpu"} here

        # 6. Mock inference engine initialization
        mock_engine_instance = MockInferenceEngine()  # Use a simple mock engine
        mock_init_engine.return_value = mock_engine_instance

        # 7. Mock inference execution
        mock_predictions = ["pred1"]  # Example predictions
        mock_run_infer.return_value = mock_predictions

        # 8. Mock evaluation execution
        # Provide a realistic structure for evaluated data
        mock_eval_results = [
            {
                "source": "q1",
                "references": ["a1"],
                "prediction": "pred1",
                "score": {
                    "global": {"accuracy": 1.0, "rougeL": 0.8},
                    "instance": {"accuracy": 1.0, "rougeL": 0.8},
                },
                "task_data": {},  # Include if expected
            }
        ]
        mock_run_eval.return_value = mock_eval_results

        # --- Act ---
        cli.main()  # Execute the main function

        # --- Assert ---
        # Verify each mocked function was called correctly

        mock_setup_parser.assert_called_once()
        mock_parser.parse_args.assert_called_once()  # Verify args parsed
        mock_setup_logging.assert_called_once_with("INFO")
        mock_prep_paths.assert_called_once_with("mock_output", "mock_results")
        mock_configure_settings.assert_called_once_with(mock_args)
        mock_context_manager.__enter__.assert_called_once()  # Check context entered
        mock_load_data.assert_called_once_with(mock_args)
        mock_prep_model_args.assert_called_once_with(mock_args)
        mock_prep_gen_kwargs.assert_called_once_with(mock_args)  # <<< Assert call

        # Assert initialize_inference_engine was called with the correct args
        # The second argument should be the combined dictionary
        mock_init_engine.assert_called_once_with(
            mock_args, combined_engine_args
        )  # <<< Key assertion

        mock_run_infer.assert_called_once_with(
            mock_engine_instance, mock_dataset_instance
        )
        mock_run_eval.assert_called_once_with(mock_predictions, mock_dataset_instance)
        # Assert process_and_save_results called with correct paths
        mock_process_save.assert_called_once_with(
            mock_args, mock_eval_results, mock_results_path, mock_samples_path
        )
        mock_context_manager.__exit__.assert_called_once()  # Check context exited
        mock_exit.assert_not_called()  # Should not exit on success

        # Check specific log messages
        mock_logger.info.assert_has_calls(
            [
                call("Starting Unitxt Evaluation CLI"),
                call("Unitxt Evaluation CLI finished successfully."),
            ],
            any_order=False,  # Set to True if order doesn't strictly matter
        )

    @patch.object(cli, "setup_parser")
    @patch.object(cli, "setup_logging")
    @patch.object(cli, "prepare_output_paths")
    @patch.object(cli, "configure_unitxt_settings")
    @patch.object(
        cli, "load_data", side_effect=FileNotFoundError("Test Not Found")
    )  # Simulate error
    @patch.object(cli, "process_and_save_results")  # Won't be called
    @patch("sys.exit")
    @patch.object(cli, "logger")
    def test_main_load_data_error(
        self,
        mock_logger,
        mock_exit,
        mock_process_save,
        mock_load_data,
        mock_configure_settings,
        mock_prep_paths,
        mock_setup_logging,
        mock_setup_parser,
    ):
        """Test main function exits correctly on FileNotFoundError during load."""
        # Arrange Mocks (similar to success case, but load_data will raise error)
        mock_parser = MagicMock()
        mock_args = argparse.Namespace(
            verbosity="INFO",
            output_path=".",
            output_file_prefix="pref",
            task="card=dummy",
            split="test",
            limit=None,
            num_fewshots=None,
            model="hf",
            model_args={},
            log_samples=False,
            trust_remote_code=False,
            disable_hf_cache=False,
            cache_dir=None,
        )
        mock_parser.parse_args.return_value = mock_args
        mock_setup_parser.return_value = mock_parser
        mock_prep_paths.return_value = ("./pref.json", "./pref_samples.json")
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = None
        mock_context_manager.__exit__.return_value = (
            None  # Important: __exit__ should still be called in finally
        )
        mock_configure_settings.return_value = mock_context_manager

        # Act
        cli.main()

        # Assertions
        mock_load_data.assert_called_once_with(mock_args)
        mock_process_save.assert_not_called()  # Should not reach saving
        # Check that the specific exception was logged
        mock_logger.exception.assert_called_once()
        self.assertIn(
            "Error loading artifact or file: Test Not Found",
            mock_logger.exception.call_args[0][0],
        )
        # Check that sys.exit(1) was called
        mock_exit.assert_called_once_with(1)
        # Check context manager exit was still called (due to try...except...finally structure implicitly)
        mock_context_manager.__exit__.assert_called_once()

    @patch.object(cli, "setup_parser")
    @patch.object(cli, "setup_logging")
    @patch.object(cli, "prepare_output_paths")
    @patch.object(cli, "configure_unitxt_settings")
    @patch.object(cli, "load_data")
    @patch.object(cli, "prepare_model_args")
    @patch.object(cli, "prepare_gen_kwargs")  # Mock this
    @patch.object(cli, "initialize_inference_engine")
    @patch.object(cli, "run_inference")
    @patch.object(cli, "run_evaluation")
    @patch.object(cli, "process_and_save_results")
    @patch("sys.exit")
    @patch.object(cli, "logger")
    def test_main_bird_remote_scenario(
        self,
        mock_logger,
        mock_exit,
        mock_process_save,
        mock_run_eval,
        mock_run_infer,
        mock_init_engine,
        mock_prep_gen_kwargs,  # <<< Added argument
        mock_prep_model_args,
        mock_load_data,
        mock_configure_settings,
        mock_prep_paths,
        mock_setup_logging,
        mock_setup_parser,
    ):
        """Test the main function simulating the BIRD remote debug config."""
        # --- Arrange Mocks ---

        # 1. Mock parser setup and args
        mock_parser = MagicMock(spec=argparse.ArgumentParser)
        # Simulate parsing of the model_args string (as if try_parse_json worked)
        parsed_model_args_input = {
            "model_name": "llama-3-3-70b-instruct",
            "max_tokens": 256,
        }
        mock_args = argparse.Namespace(
            task="card=cards.text2sql.bird,template=templates.text2sql.you_are_given_with_hint_with_sql_prefix",
            model="cross_provider",  # Use remote model type
            model_args=parsed_model_args_input,  # Args as parsed dict
            split="validation",
            limit=100,
            num_fewshots=None,
            gen_kwargs=None,  # Explicitly None
            output_path="./debug_output/bird_remote",
            output_file_prefix="evaluation_results",
            log_samples=True,
            verbosity="INFO",
            trust_remote_code=True,  # Specific to this scenario
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

        # 3. Mock context manager
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = None
        mock_context_manager.__exit__.return_value = None
        mock_configure_settings.return_value = mock_context_manager

        # 4. Mock data loading
        mock_dataset_instance = MockHFDataset(
            [{"source": f"q{i}"} for i in range(2)]
        )  # 2 samples
        self.assertEqual(len(mock_dataset_instance), 2)  # Verify length
        mock_load_data.return_value = mock_dataset_instance

        # 5. Mock argument preparation
        # prepare_model_args should just return the already parsed dict in this case
        mock_prep_model_args.return_value = parsed_model_args_input

        # Mock prepare_gen_kwargs (returns {} if args.gen_kwargs is None)
        mock_gen_kwargs_prepared = {}
        mock_prep_gen_kwargs.return_value = (
            mock_gen_kwargs_prepared  # <<< Set return value
        )

        # Calculate the combined dict passed to initialize_inference_engine
        combined_engine_args = parsed_model_args_input.copy()
        combined_engine_args.update(
            mock_gen_kwargs_prepared
        )  # Still parsed_model_args_input

        # 6. Mock engine initialization (using CrossProvider mock)
        mock_remote_engine_instance = MockCrossProviderInferenceEngine()
        mock_init_engine.return_value = mock_remote_engine_instance

        # 7. Mock inference
        mock_predictions = ["sql pred 1", "sql pred 2"]  # Match dataset length
        mock_run_infer.return_value = mock_predictions

        # 8. Mock evaluation
        mock_eval_results = [
            {
                "source": "q0",
                "prediction": "sql pred 1",
                "score": {"global": {"accuracy": 0.5}, "instance": {"accuracy": 0.0}},
            },
            {
                "source": "q1",
                "prediction": "sql pred 2",
                "score": {"global": {"accuracy": 0.5}, "instance": {"accuracy": 1.0}},
            },
        ]  # Example results matching dataset length
        mock_run_eval.return_value = mock_eval_results

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
        # Check specific args passed to configure_settings if needed
        self.assertTrue(mock_configure_settings.call_args[0][0].trust_remote_code)
        mock_context_manager.__enter__.assert_called_once()
        mock_load_data.assert_called_once_with(mock_args)
        # Check args passed to load_data (limit applied) - access via call_args
        # call_args is a tuple: (positional_args, keyword_args)
        # positional_args is a tuple, so access the first element (mock_args)
        # then access the attribute 'limit'
        # self.assertEqual(mock_load_data.call_args[0][0].limit, 100) # This checks the Namespace passed
        # self.assertIsNone(mock_load_data.call_args[0][0].num_fewshots) # This checks the Namespace passed

        mock_prep_model_args.assert_called_once_with(mock_args)
        mock_prep_gen_kwargs.assert_called_once_with(mock_args)  # <<< Assert call

        # Assert initialize_inference_engine with the combined dict
        mock_init_engine.assert_called_once_with(
            mock_args, combined_engine_args
        )  # <<< Corrected assertion

        # Check specific args passed to init_engine if needed
        self.assertEqual(mock_init_engine.call_args[0][0].model, "cross_provider")
        mock_run_infer.assert_called_once_with(
            mock_remote_engine_instance, mock_dataset_instance
        )
        mock_run_eval.assert_called_once_with(mock_predictions, mock_dataset_instance)
        mock_process_save.assert_called_once_with(
            mock_args, mock_eval_results, expected_results_path, expected_samples_path
        )
        # Check specific args passed to process_save if needed
        self.assertTrue(mock_process_save.call_args[0][0].log_samples)
        mock_context_manager.__exit__.assert_called_once()
        mock_exit.assert_not_called()
        mock_logger.info.assert_has_calls(
            [
                call("Starting Unitxt Evaluation CLI"),
                # Add more specific log calls if needed for this scenario
                call("Unitxt Evaluation CLI finished successfully."),
            ],
            any_order=False,  # Usually check order unless irrelevant
        )


if __name__ == "__main__":
    unittest.main()
