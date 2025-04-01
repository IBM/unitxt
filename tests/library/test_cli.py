import argparse
import os
import unittest
from unittest.mock import MagicMock, call, mock_open, patch

from unitxt import evaluate_cli as cli


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
        )
        # Check stripping (assuming original code strips key but not value)
        self.assertEqual(cli._parse_key_value_string(""), None)
        self.assertEqual(cli._parse_key_value_string("key_no_value"), None)
        # Check logging on invalid part (use assertLogs)
        with self.assertLogs(cli.logger, level="WARNING"):
            result = cli._parse_key_value_string("key=val1,invalid,key2=val2")
        self.assertEqual(result, {"key": "val1", "key2": "val2"})

    def test_try_parse_json_valid(self):
        """Test try_parse_json with valid JSON and key-value strings."""
        self.assertEqual(
            cli.try_parse_json('{"key": "value", "num": 1}'), {"key": "value", "num": 1}
        )
        self.assertEqual(
            cli.try_parse_json("key=value,num=1"), {"key": "value", "num": 1}
        )
        self.assertEqual(cli.try_parse_json("just a string"), "just a string")
        self.assertIsNone(cli.try_parse_json(None))

    def test_try_parse_json_invalid(self):
        """Test try_parse_json with invalid JSON."""
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "Invalid JSON"):
            cli.try_parse_json('{key: "value"}')
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "Invalid JSON"):
            cli.try_parse_json("[1, 2")

    def test_setup_parser(self):
        """Test if the argument parser is set up with expected arguments."""
        parser = cli.setup_parser()
        actions = {action.dest: action for action in parser._actions}
        self.assertIn("tasks", actions)
        self.assertTrue(actions["tasks"].required)
        self.assertIn("model", actions)
        self.assertEqual(actions["model"].default, "hf")
        self.assertIn("model_args", actions)
        self.assertEqual(actions["model_args"].type, cli.try_parse_json)
        self.assertIn("output_dir", actions)
        self.assertEqual(actions["output_dir"].default, ".")
        self.assertIn("log_samples", actions)
        self.assertFalse(actions["log_samples"].default)
        self.assertIn("verbosity", actions)
        self.assertEqual(actions["verbosity"].default, "INFO")
        self.assertIn("trust_remote_code", actions)
        self.assertFalse(actions["trust_remote_code"].default)

    @patch("os.makedirs")
    def test_prepare_output_paths(self, mock_makedirs):
        """Test output path generation and directory creation."""
        results_path, samples_path = cli.prepare_output_paths(
            "/tmp/output", "my_results"
        )
        mock_makedirs.assert_called_once_with("/tmp/output", exist_ok=True)
        self.assertEqual(results_path, "/tmp/output/my_results.json")
        self.assertEqual(samples_path, "/tmp/output/my_results_samples.json")

    # Added back test for settings config
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

        self.assertEqual(context, mock_context_manager)
        mock_settings.context.assert_called_once_with(
            disable_hf_datasets_cache=True,
            allow_unverified_code=True,
            hf_cache_dir="/path/to/cache",
        )
        self.assertEqual(os.environ.get("HF_DATASETS_CACHE"), "/path/to/cache")
        # Reset environ for other tests
        if "HF_DATASETS_CACHE" in os.environ:
            del os.environ["HF_DATASETS_CACHE"]

    # --- Data Loading Tests ---
    @patch.object(cli, "load_dataset")
    def test_load_data(self, mock_load_dataset):
        """Test data loading function."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_load_dataset.return_value = mock_dataset

        args = argparse.Namespace(
            tasks="card=c,template=t", split="validation", limit=5
        )
        with patch.object(cli, "logger"):  # Mock logger
            dataset = cli.load_data(args)

        self.assertEqual(dataset, mock_dataset)
        mock_load_dataset.assert_called_once_with(
            "card=c,template=t,loader_limit=5", split="validation"
        )
        self.assertEqual(len(dataset), 10)  # Check if len is called

    @patch.object(cli, "load_dataset")
    def test_load_data_limit_in_task_string(self, mock_load_dataset):
        """Test data loading when limit is already in the task string."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 7
        mock_load_dataset.return_value = mock_dataset

        args = argparse.Namespace(
            tasks="card=c,template=t,loader_limit=7",
            split="test",
            limit=5,  # --limit should override
        )
        with self.assertLogs(cli.logger, level="WARNING") as cm:
            dataset = cli.load_data(args)
        self.assertIn(
            "Limit specified in both --tasks string and --limit arg", cm.output[0]
        )

        self.assertEqual(dataset, mock_dataset)
        # Check that the limit from --limit (5) overrides the one in the string (7)
        mock_load_dataset.assert_called_once_with(
            "card=c,template=t,loader_limit=5", split="test"
        )

    # --- Model Arg Prep ---
    def test_prepare_model_args(self):
        """Test model argument preparation."""
        args_dict = argparse.Namespace(model_args={"key": "value", "num": 1})
        with patch.object(cli, "logger"):  # Mock logger
            res = cli.prepare_model_args(args_dict)
        self.assertEqual(res, {"key": "value", "num": 1})

        # Test with parsed key-value string
        args_dict_parsed = argparse.Namespace(
            model_args=cli.try_parse_json("key=value,num=1")
        )
        with patch.object(cli, "logger"):  # Mock logger
            res_parsed = cli.prepare_model_args(args_dict_parsed)
        self.assertEqual(res_parsed, {"key": "value", "num": 1})

        args_empty = argparse.Namespace(model_args={})
        with patch.object(cli, "logger"):  # Mock logger
            res_empty = cli.prepare_model_args(args_empty)
        self.assertEqual(res_empty, {})

    # --- Inference Engine Initialization Tests ---

    @patch.object(cli, "_package_is_available")
    @patch.object(cli, "HFAutoModelInferenceEngine")
    def test_initialize_inference_engine_hf_success(
        self, mock_hf_engine, mock_pkg_avail
    ):
        """Test initializing HF engine successfully."""
        mock_pkg_avail.return_value = True
        args = argparse.Namespace(model="hf")
        model_args_dict = {
            "pretrained": "model_id",
            "device": "cuda:0",
            "torch_dtype": "bfloat16",
        }
        with patch.object(cli, "logger"):
            engine = cli.initialize_inference_engine(args, model_args_dict.copy())

        self.assertIsInstance(engine, MagicMock)
        mock_hf_engine.assert_called_once_with(
            model_name="model_id", device="cuda:0", torch_dtype="bfloat16"
        )

    @patch.object(cli, "_package_is_available")
    @patch.object(cli, "HFAutoModelInferenceEngine")
    def test_initialize_inference_engine_hf_missing_pretrained(
        self, mock_hf_engine, mock_pkg_avail
    ):
        """Test initializing HF engine with missing 'pretrained' arg."""
        mock_pkg_avail.return_value = True
        args = argparse.Namespace(model="hf")
        model_args_dict = {"device": "cpu"}  # Missing 'pretrained'

        with self.assertRaisesRegex(ValueError, "Argument 'pretrained' is required"):
            with self.assertLogs(cli.logger, level="ERROR"):
                cli.initialize_inference_engine(args, model_args_dict)

        mock_hf_engine.assert_not_called()

    @patch.object(cli, "_package_is_available")
    @patch.object(cli, "CrossProviderInferenceEngine")
    def test_initialize_inference_engine_remote_success(
        self, mock_remote_engine, mock_pkg_avail
    ):
        """Test initializing remote engine successfully."""
        mock_pkg_avail.return_value = True
        args = argparse.Namespace(model="generic_remote")
        model_args_dict = {
            "model_name": "openai/gpt-4o",
            "max_tokens": 100,
            "temperature": 0.5,
        }
        with patch.object(cli, "logger"):
            engine = cli.initialize_inference_engine(args, model_args_dict.copy())

        self.assertIsInstance(engine, MagicMock)
        mock_remote_engine.assert_called_once_with(
            model="openai/gpt-4o", max_tokens=100, temperature=0.5
        )
        self.assertEqual(
            mock_remote_engine.call_args[1],
            {"max_tokens": 100, "model": "openai/gpt-4o", "temperature": 0.5},
        )

    # Added test for missing remote model_name arg
    @patch.object(cli, "_package_is_available")
    @patch.object(cli, "CrossProviderInferenceEngine")
    def test_initialize_inference_engine_remote_missing_model_name(
        self, mock_remote_engine, mock_pkg_avail
    ):
        """Test initializing remote engine with missing 'model_name' arg."""
        mock_pkg_avail.return_value = True
        args = argparse.Namespace(model="generic_remote")
        model_args_dict = {"max_tokens": 100}  # Missing 'model_name'

        with self.assertRaisesRegex(ValueError, "Argument 'model_name' is required"):
            with self.assertLogs(cli.logger, level="ERROR"):
                cli.initialize_inference_engine(args, model_args_dict)
        mock_remote_engine.assert_not_called()

    # --- Inference Execution ---
    @patch.object(
        cli.InferenceEngine, "infer"
    )  # Patch base class method maybe? Or mock engine instance.
    def test_run_inference_success(self, mock_infer_method):
        """Test inference runner success."""
        mock_engine_instance = MagicMock(spec=cli.InferenceEngine)
        # Configure the mock 'infer' method on the instance
        mock_engine_instance.infer.return_value = ["pred1", "pred2"]
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 2  # Match prediction count

        with patch.object(cli, "logger"):
            predictions = cli.run_inference(mock_engine_instance, mock_dataset)

        self.assertEqual(predictions, ["pred1", "pred2"])
        mock_engine_instance.infer.assert_called_once_with(mock_dataset)

    @patch.object(cli.InferenceEngine, "infer")
    def test_run_inference_failure(self, mock_infer_method):
        """Test inference runner failure."""
        mock_engine_instance = MagicMock(spec=cli.InferenceEngine)
        mock_engine_instance.infer.side_effect = ConnectionError("API failed")
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 1

        with self.assertRaises(ConnectionError):
            with patch.object(cli, "logger") as mock_logger:
                cli.run_inference(mock_engine_instance, mock_dataset)
                mock_logger.exception.assert_called_once()  # Check exception log

    # --- Evaluation Execution ---
    @patch.object(cli, "evaluate")
    def test_run_evaluation_success(self, mock_evaluate):
        """Test the evaluation runner successfully."""
        mock_predictions = ["pred1", "pred2"]
        mock_dataset = MagicMock()
        mock_eval_results = [{"score": {"global": {}, "instance": {}}}]
        mock_evaluate.return_value = mock_eval_results
        with patch.object(cli, "logger"):
            results = cli.run_evaluation(mock_predictions, mock_dataset)
        self.assertEqual(results, mock_eval_results)
        mock_evaluate.assert_called_once_with(
            predictions=mock_predictions, data=mock_dataset
        )

    @patch.object(cli, "evaluate")
    def test_run_evaluation_failure(self, mock_evaluate):
        """Test the evaluation runner when evaluate fails."""
        mock_predictions = ["pred1"]
        mock_dataset = MagicMock()
        mock_evaluate.side_effect = ValueError("Evaluation failed")
        with self.assertRaises(ValueError):
            with patch.object(cli, "logger") as mock_logger:
                cli.run_evaluation(mock_predictions, mock_dataset)
                mock_logger.exception.assert_called_once()

    # --- Result Processing/Saving Tests ---

    @patch.object(cli, "logger")
    def test_extract_scores_and_samples(self, mock_logger):
        """Test extracting scores and samples from evaluated data."""
        evaluated_data = [
            {
                "source": "s1",
                "prediction": "p1",
                "references": ["r1"],
                "score": {"global": {"acc": 1.0, "f1": 0.9}, "instance": {"acc": 1.0}},
                "task_data": {"id": 1},
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
        self.assertEqual(global_scores, {"acc": 1.0, "f1": 0.9})
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]["metrics"], {"acc": 1.0})
        mock_logger.warning.assert_not_called()

    @patch.object(cli, "logger")
    def test_extract_scores_and_samples_no_global(self, mock_logger):
        """Test extraction when global scores are missing."""
        evaluated_data = [{"score": {"instance": {"acc": 1.0}}}]
        global_scores, samples = cli._extract_scores_and_samples(evaluated_data)
        self.assertEqual(global_scores, {})
        self.assertEqual(len(samples), 1)
        mock_logger.warning.assert_called_once()
        self.assertIn(
            "Could not automatically locate global scores",
            mock_logger.warning.call_args[0][0],
        )

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    @patch.object(cli, "logger")
    def test_save_results_to_disk_summary_only(
        self, mock_logger, mock_json_dump, mock_file_open
    ):
        """Test saving results to disk (summary only)."""
        args = argparse.Namespace(log_samples=False, tasks="task1", limit=10)
        global_scores = {"acc": 1.0}
        samples_data = [{"index": 0}]
        results_path, samples_path = "/path/res.json", "/path/samp.json"
        cli._save_results_to_disk(
            args, global_scores, samples_data, results_path, samples_path
        )
        mock_file_open.assert_called_once_with(results_path, "w", encoding="utf-8")
        expected_summary = {"config": vars(args), "global_scores": global_scores}
        mock_json_dump.assert_called_once_with(
            expected_summary, mock_file_open(), indent=4, ensure_ascii=False
        )
        mock_logger.info.assert_called_once_with(
            f"Saving global results summary to: {results_path}"
        )

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    @patch.object(cli, "logger")
    def test_save_results_to_disk_with_samples(
        self, mock_logger, mock_json_dump, mock_file_open
    ):
        """Test saving results to disk including samples."""
        args = argparse.Namespace(log_samples=True, tasks="task1", limit=10)
        global_scores = {"acc": 1.0}
        samples_data = [{"index": 0}]
        results_path, samples_path = "/path/res.json", "/path/samp.json"
        cli._save_results_to_disk(
            args, global_scores, samples_data, results_path, samples_path
        )
        calls = [
            call(results_path, "w", encoding="utf-8"),
            call(samples_path, "w", encoding="utf-8"),
        ]
        mock_file_open.assert_has_calls(calls, any_order=True)
        self.assertEqual(mock_json_dump.call_count, 2)
        log_calls = [
            call(f"Saving global results summary to: {results_path}"),
            call(f"Saving detailed samples to: {samples_path}"),
        ]
        mock_logger.info.assert_has_calls(log_calls, any_order=True)

    # Added test for save error
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump", side_effect=OSError("Disk full"))
    @patch.object(cli, "logger")
    def test_save_results_to_disk_error(
        self, mock_logger, mock_json_dump, mock_file_open
    ):
        """Test saving results when json.dump fails."""
        args = argparse.Namespace(log_samples=False, tasks="task1")
        global_scores = {"acc": 1.0}
        samples_data = []
        results_path, samples_path = "res.json", "samp.json"
        # Exception expected from process_and_save_results if _save fails
        with self.assertRaises(IOError):
            cli._save_results_to_disk(
                args, global_scores, samples_data, results_path, samples_path
            )

    @patch.object(cli, "print_dict")
    @patch.object(cli, "_extract_scores_and_samples")
    @patch.object(cli, "_save_results_to_disk")
    @patch.object(cli, "logger")
    def test_process_and_save_results(
        self, mock_logger, mock_save, mock_extract, mock_print_dict
    ):
        """Test the main result processing and saving function."""
        args = MagicMock(log_samples=True)
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

        cli.process_and_save_results(
            args, evaluated_dataset, results_path, samples_path
        )

        mock_extract.assert_called_once_with(evaluated_dataset)
        mock_print_dict.assert_has_calls(
            [
                call(mock_global_scores),
                call(mock_samples_data[0], keys_to_print=unittest.mock.ANY),
            ]
        )
        mock_save.assert_called_once_with(
            args, mock_global_scores, mock_samples_data, results_path, samples_path
        )
        mock_logger.info.assert_has_calls(
            [
                call("\n--- Global Scores ---"),
                call("\n--- Example Instance (Index 0) ---"),
            ]
        )

    @patch.object(cli, "setup_parser")
    @patch.object(cli, "setup_logging")
    @patch.object(cli, "prepare_output_paths")
    @patch.object(cli, "configure_unitxt_settings")
    @patch.object(cli, "load_data")
    @patch.object(cli, "prepare_model_args")
    @patch.object(cli, "initialize_inference_engine")
    @patch.object(cli, "run_inference")
    @patch.object(cli, "run_evaluation")
    @patch.object(cli, "process_and_save_results")
    @patch("sys.exit")
    @patch.object(cli, "logger")
    def test_main_success_flow(
        self,
        mock_logger,
        mock_exit,
        mock_process_save,
        mock_run_eval,
        mock_run_infer,
        mock_init_engine,
        mock_prep_model_args,
        mock_load_data,
        mock_configure_settings,
        mock_prep_paths,
        mock_setup_logging,
        mock_setup_parser,
    ):
        """Test the main function success path."""
        # Arrange Mocks
        mock_parser = MagicMock()
        # --- FIX IS HERE: Add missing attributes to mock_args ---
        mock_args = argparse.Namespace(
            # Existing ones
            verbosity="INFO",
            output_dir=".",
            output_file_prefix="pref",
            # Add missing ones based on setup_parser defaults
            tasks="card=dummy",  # Required, give dummy value
            split="test",
            limit=None,
            model="hf",
            model_args={},  # <<< The missing attribute
            log_samples=False,
            trust_remote_code=False,
            disable_hf_cache=False,
            cache_dir=None,
        )
        # --- End of Fix ---
        mock_parser.parse_args.return_value = mock_args
        mock_setup_parser.return_value = mock_parser
        mock_prep_paths.return_value = ("./pref.json", "./pref_samples.json")
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = None
        mock_context_manager.__exit__.return_value = None
        mock_configure_settings.return_value = mock_context_manager
        mock_dataset = MagicMock()
        mock_load_data.return_value = mock_dataset
        # We added model_args to mock_args, so prepare_model_args will receive it
        mock_model_args_dict = {}  # Corresponds to default model_args={}
        mock_prep_model_args.return_value = mock_model_args_dict
        mock_engine = MagicMock()
        mock_init_engine.return_value = mock_engine
        mock_predictions = ["p1"]
        mock_run_infer.return_value = mock_predictions
        mock_eval_results = [{"score": {}}]
        mock_run_eval.return_value = mock_eval_results

        # Act
        cli.main()

        # Assert basic flow (should pass now if args are correct)
        mock_setup_parser.assert_called_once()
        mock_parser.parse_args.assert_called_once()
        mock_setup_logging.assert_called_once_with("INFO")
        # Assert prepare_output_paths was called with correct args from mock_args
        mock_prep_paths.assert_called_once_with(".", "pref")
        # Assert configure_unitxt_settings was called with the full mock_args
        mock_configure_settings.assert_called_once_with(mock_args)
        mock_context_manager.__enter__.assert_called_once()
        # Assert load_data was called with the full mock_args
        mock_load_data.assert_called_once_with(mock_args)
        # Assert prepare_model_args was called with the full mock_args
        mock_prep_model_args.assert_called_once_with(mock_args)
        # Assert initialize_inference_engine with full mock_args and result of prepare_model_args
        mock_init_engine.assert_called_once_with(mock_args, mock_model_args_dict)
        # Assert run_inference with engine and dataset
        mock_run_infer.assert_called_once_with(mock_engine, mock_dataset)
        # Assert run_evaluation with predictions and dataset
        mock_run_eval.assert_called_once_with(mock_predictions, mock_dataset)
        # Assert process_and_save_results with full args, eval results, and paths
        mock_process_save.assert_called_once_with(
            mock_args, mock_eval_results, "./pref.json", "./pref_samples.json"
        )
        mock_context_manager.__exit__.assert_called_once()
        mock_exit.assert_not_called()
        mock_logger.info.assert_has_calls(
            [
                call("Starting Unitxt Evaluation CLI"),
                call("Unitxt Evaluation CLI finished."),
            ]
        )

    # Simplified main test focusing on specific error exit
    @patch.object(cli, "setup_parser")
    @patch.object(cli, "setup_logging")
    @patch.object(cli, "prepare_output_paths")
    @patch.object(cli, "configure_unitxt_settings")
    @patch.object(
        cli, "load_data", side_effect=FileNotFoundError("Test Not Found")
    )  # Error source
    @patch.object(
        cli, "process_and_save_results"
    )  # Need to mock downstream to prevent execution
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
        mock_parser = MagicMock()
        # --- FIX IS HERE: Add missing attributes to mock_args ---
        mock_args = argparse.Namespace(
            # Existing ones
            verbosity="INFO",
            output_dir=".",
            output_file_prefix="pref",
            # Add missing ones based on setup_parser defaults
            tasks="card=dummy",  # Required, give dummy value
            split="test",
            limit=None,
            model="hf",
            model_args={},  # <<< The missing attribute
            log_samples=False,
            trust_remote_code=False,
            disable_hf_cache=False,
            cache_dir=None,
        )
        # --- End of Fix ---
        mock_parser.parse_args.return_value = mock_args
        mock_setup_parser.return_value = mock_parser
        mock_prep_paths.return_value = ("./pref.json", "./pref_samples.json")
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = None
        mock_context_manager.__exit__.return_value = None
        mock_configure_settings.return_value = mock_context_manager

        cli.main()  # Call main after setting up mocks

        # Assertions
        mock_load_data.assert_called_once_with(mock_args)  # Check load_data was called
        mock_process_save.assert_not_called()  # Should not be reached
        mock_logger.exception.assert_called_once()  # Check error was logged via exception
        # Check the content of the logged exception message
        self.assertIn(
            "Error during setup or data processing: Test Not Found",
            mock_logger.exception.call_args[0][0],
        )
        mock_exit.assert_called_once_with(1)  # Check sys.exit(1) was called

    # --- Test for Specific Debug Scenario ---

    @patch.object(cli, "setup_parser")
    @patch.object(cli, "setup_logging")
    @patch.object(cli, "prepare_output_paths")
    @patch.object(cli, "configure_unitxt_settings")
    @patch.object(cli, "load_data")
    @patch.object(cli, "prepare_model_args")
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
        mock_prep_model_args,
        mock_load_data,
        mock_configure_settings,
        mock_prep_paths,
        mock_setup_logging,
        mock_setup_parser,
    ):
        """Test the main function simulating the BIRD remote debug config."""
        # Arrange Mocks
        mock_parser = MagicMock()

        # Create the args namespace reflecting the debug configuration
        # Use try_parse_json to handle the model_args string like argparse would
        parsed_model_args = cli.try_parse_json(
            "model_name=llama-3-3-70b-instruct,max_tokens=256"
        )
        mock_args = argparse.Namespace(
            tasks="card=cards.text2sql.bird,template=templates.text2sql.you_are_given_with_hint_with_sql_prefix",
            model="generic_remote",
            model_args=parsed_model_args,  # Use the parsed dict
            split="validation",
            limit=100,
            output_dir="./debug_output/bird_remote",
            log_samples=True,  # Flag present means True
            verbosity="INFO",
            trust_remote_code=True,  # Flag present means True
            # Defaults for args not specified in debug config
            output_file_prefix="evaluation_results",
            disable_hf_cache=False,
            cache_dir=None,
        )

        mock_parser.parse_args.return_value = mock_args
        mock_setup_parser.return_value = mock_parser

        # Set up return values for mocked functions
        expected_results_path = "./debug_output/bird_remote/evaluation_results.json"
        expected_samples_path = (
            "./debug_output/bird_remote/evaluation_results_samples.json"
        )
        mock_prep_paths.return_value = (expected_results_path, expected_samples_path)

        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = None
        mock_context_manager.__exit__.return_value = None
        mock_configure_settings.return_value = mock_context_manager

        mock_dataset = MagicMock()
        mock_load_data.return_value = mock_dataset

        # prepare_model_args should just return the already parsed dict from mock_args
        expected_model_args_dict = parsed_model_args
        mock_prep_model_args.return_value = expected_model_args_dict

        # Mock the specific engine expected
        mock_remote_engine_instance = MagicMock(spec=cli.CrossProviderInferenceEngine)
        mock_init_engine.return_value = mock_remote_engine_instance

        mock_predictions = ["sql pred 1", "sql pred 2"]  # Dummy predictions
        mock_run_infer.return_value = mock_predictions

        mock_eval_results = [{"score": {"global": {"accuracy": 0.5}}}]  # Dummy results
        mock_run_eval.return_value = mock_eval_results

        # Act
        cli.main()

        # Assert flow with specific arguments
        mock_setup_parser.assert_called_once()
        mock_parser.parse_args.assert_called_once()
        mock_setup_logging.assert_called_once_with("INFO")  # From mock_args
        # Check paths based on mock_args
        mock_prep_paths.assert_called_once_with(
            "./debug_output/bird_remote", "evaluation_results"
        )
        # Check settings based on mock_args (specifically trust_remote_code)
        mock_configure_settings.assert_called_once_with(mock_args)
        self.assertTrue(mock_configure_settings.call_args[0][0].trust_remote_code)
        mock_context_manager.__enter__.assert_called_once()
        # Check load_data called with the specific mock_args
        mock_load_data.assert_called_once_with(mock_args)
        # Check prepare_model_args called with specific mock_args
        mock_prep_model_args.assert_called_once_with(mock_args)
        # Check initialize_inference_engine called correctly for remote model
        mock_init_engine.assert_called_once_with(mock_args, expected_model_args_dict)
        # Check run_inference called with the mocked remote engine instance
        mock_run_infer.assert_called_once_with(
            mock_remote_engine_instance, mock_dataset
        )
        # Check run_evaluation called correctly
        mock_run_eval.assert_called_once_with(mock_predictions, mock_dataset)
        # Check process_and_save_results called correctly (paths and log_samples)
        mock_process_save.assert_called_once_with(
            mock_args, mock_eval_results, expected_results_path, expected_samples_path
        )
        self.assertTrue(
            mock_process_save.call_args[0][0].log_samples
        )  # Verify log_samples=True was passed
        mock_context_manager.__exit__.assert_called_once()
        mock_exit.assert_not_called()  # Ensure successful execution
        mock_logger.info.assert_has_calls(
            [
                call("Starting Unitxt Evaluation CLI"),
                call("Unitxt Evaluation CLI finished."),
            ]
        )


if __name__ == "__main__":
    unittest.main()
