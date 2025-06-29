from unitxt.error_utils import Documentation, UnitxtError, UnitxtWarning, error_context

from tests.utils import UnitxtTestCase


class TestErrorUtils(UnitxtTestCase):
    def test_error_no_additional_info(self):
        with self.assertRaises(UnitxtError) as e:
            raise UnitxtError("This should fail")
        self.assertEqual(str(e.exception), "This should fail")

    def test_error_with_additional_info(self):
        with self.assertRaises(UnitxtError) as e:
            raise UnitxtError("This should fail", Documentation.ADDING_TASK)
        self.assertEqual(
            str(e.exception),
            "This should fail\nFor more information: see https://www.unitxt.ai/en/latest//docs/adding_task.html \n",
        )

    def test_warning_no_additional_info(self):
        UnitxtWarning("This should fail")

    def test_warning_with_additional_info(self):
        UnitxtWarning("This should fail", Documentation.ADDING_TASK)


class TestContextEnhancedErrors(UnitxtTestCase):
    def test_error_context_with_original_error_only(self):
        """Test error_context with just the original error."""
        with self.assertRaises(ValueError) as cm:
            with error_context():
                raise ValueError("Original error message")

        error = cm.exception
        # With version info, the message now includes a context box
        message = str(error)
        self.assertIn("Original error message", message)
        self.assertIn("Unitxt Error Context", message)
        self.assertIn("Unitxt:", message)
        self.assertIn("Python:", message)

    def test_error_context_with_context_object(self):
        """Test error_context with a context object."""

        class TestProcessor:
            pass

        processor = TestProcessor()

        with self.assertRaises(KeyError) as cm:
            with error_context(processor):
                raise KeyError("Missing key")

        error = cm.exception
        # KeyError has special __str__ behavior that adds quotes, so we check the content is there
        message = str(error)
        self.assertIn("Error Context", message)
        self.assertIn("Object: TestProcessor", message)
        self.assertIn("Missing key", message)
        # Check for box characters
        self.assertIn("┌", message)
        self.assertIn("└", message)

    def test_error_context_with_context_fields(self):
        """Test error_context with custom context fields."""
        with self.assertRaises(RuntimeError) as cm:
            with error_context(
                file_name="data.json", line_number=42, operation="parse"
            ):
                raise RuntimeError("Processing failed")

        error = cm.exception

        message = str(error)
        self.assertIn("Error Context", message)
        self.assertIn("File Name: data.json", message)
        self.assertIn("Line Number: 42", message)
        self.assertIn("Operation: parse", message)
        self.assertIn("Processing failed", message)
        # Check for box characters
        self.assertIn("┌", message)
        self.assertIn("└", message)

    def test_error_context_with_object_and_context(self):
        """Test error_context with both context object and context fields."""

        class DatabaseConnection:
            pass

        db = DatabaseConnection()

        with self.assertRaises(ConnectionError) as cm:
            with error_context(db, table="users", operation="SELECT", batch_size=1000):
                raise ConnectionError("Connection timeout")

        error = cm.exception
        message = str(error)
        self.assertIn("Error Context", message)
        self.assertIn("Object: DatabaseConnection", message)
        self.assertIn("Batch Size: 1000", message)
        self.assertIn("Operation: SELECT", message)
        self.assertIn("Table: users", message)
        self.assertIn("Connection timeout", message)
        # Check for box characters
        self.assertIn("┌", message)
        self.assertIn("└", message)

    def test_error_context_with_none_values(self):
        """Test error_context handles None values correctly."""
        with self.assertRaises(ValueError) as cm:
            with error_context(stream_name=None, valid_field="test_value"):
                raise ValueError("Test error")

        error = cm.exception
        message = str(error)
        self.assertIn("Error Context", message)
        self.assertIn("Stream Name: unknown", message)
        self.assertIn("Valid Field: test_value", message)
        self.assertIn("Test error", message)
        # Check for box characters
        self.assertIn("┌", message)
        self.assertIn("└", message)

    def test_error_context_field_name_formatting(self):
        """Test that field names are properly formatted."""
        with self.assertRaises(ValueError) as cm:
            with error_context(
                field_name="input_text",
                processing_step="tokenization",
                model_name="bert_base",
                gpu_device="cuda_0",
            ):
                raise ValueError("Test error")

        error = cm.exception
        # Check that underscores are replaced with spaces and titles are applied
        message = str(error)
        self.assertIn("Field Name: input_text", message)
        self.assertIn("Processing Step: tokenization", message)
        self.assertIn("Model Name: bert_base", message)
        self.assertIn("Gpu Device: cuda_0", message)

    def test_error_context_original_error_access(self):
        """Test that original error is accessible via attributes."""
        original_error = ValueError("Original error")

        with self.assertRaises(ValueError) as cm:
            with error_context():
                raise original_error

        error = cm.exception
        # The original_error attribute should have the same type and message
        self.assertIsInstance(error.original_error, ValueError)
        self.assertEqual(str(error.original_error), "Original error")


class TestErrorContext(UnitxtTestCase):
    def test_error_context_no_error(self):
        """Test error_context when no error occurs."""
        with error_context("test_object", operation="test"):
            result = "success"

        self.assertEqual(result, "success")

    def test_error_context_with_error(self):
        """Test error_context when an error occurs."""

        class TestProcessor:
            pass

        processor = TestProcessor()

        with self.assertRaises(ValueError) as cm:
            with error_context(processor, operation="validation", item_id=42):
                raise ValueError("Something went wrong")

        error = cm.exception
        self.assertIsInstance(error, ValueError)
        self.assertEqual(str(error.original_error), "Something went wrong")
        self.assertEqual(error.context_object, processor)
        # Context now includes version info plus the specified context
        self.assertIn("Unitxt", error.context)
        self.assertIn("Python", error.context)
        self.assertEqual(error.context["operation"], "validation")
        self.assertEqual(error.context["item_id"], 42)

    def test_error_context_without_object(self):
        """Test error_context without a context object."""
        with self.assertRaises(KeyError) as cm:
            with error_context(input_file="data.json", line_number=156):
                raise KeyError("Missing field")

        error = cm.exception
        self.assertIsInstance(error, KeyError)
        self.assertIsNone(error.context_object)
        # Context now includes version info plus the specified context
        self.assertIn("Unitxt", error.context)
        self.assertIn("Python", error.context)
        self.assertEqual(error.context["input_file"], "data.json")
        self.assertEqual(error.context["line_number"], 156)

    def test_error_context_preserves_original_traceback(self):
        """Test that error_context preserves the original exception's traceback."""
        caught_error = None

        try:
            with error_context(operation="test"):
                raise ValueError("Original error")  # This is line 180
        except ValueError as e:
            caught_error = e

        self.assertIsNotNone(caught_error)
        self.assertIsInstance(caught_error, ValueError)
        self.assertEqual(str(caught_error.original_error), "Original error")

        # Test that the traceback is preserved and points to the original raise location
        self.assertIsNotNone(caught_error.__traceback__)
        # The traceback should show the line number where the error was processed
        # The traceback points to the raise statement where the original error was raised
        self.assertEqual(
            caught_error.__traceback__.tb_lineno, 208
        )  # Line where raise ValueError occurs

    def test_error_context_nested_contexts(self):
        """Test nested error_context calls."""

        class OuterProcessor:
            pass

        class InnerProcessor:
            pass

        outer = OuterProcessor()
        inner = InnerProcessor()

        # In nested contexts, the outer context catches and re-raises the inner error
        with self.assertRaises(RuntimeError) as cm:
            with error_context(outer, stage="outer", step=1):
                with error_context(inner, stage="inner", step=2, index=1):
                    raise RuntimeError("Nested error")

        # Should preserve the innermost context object (where the error actually occurred)
        # but combine context data from both levels (outer context overrides inner when keys conflict)
        error = cm.exception
        self.assertEqual(error.context_object, inner)
        # expected_context = {
        #     "Unitxt": "1.24.0",  # This might vary by environment
        #     "Python": "3.10.18",  # This might vary by environment
        #     "stage": "outer",
        #     "step": 1,
        #     "index": 1,
        # }
        # Check that the context contains the expected keys and values for non-version fields
        self.assertIn("Unitxt", error.context)
        self.assertIn("Python", error.context)
        self.assertEqual(error.context["stage"], "outer")
        self.assertEqual(error.context["step"], 1)
        self.assertEqual(error.context["index"], 1)

        # The error message should show the innermost context object but merged context data
        message = str(error)
        self.assertIn("Object: InnerProcessor", message)
        self.assertIn("Stage: outer", message)  # outer context overrides inner

    def test_error_context_empty_context(self):
        """Test error_context with empty context."""
        with self.assertRaises(ValueError) as cm:
            with error_context():
                raise ValueError("Test error")

        error = cm.exception
        self.assertIsNone(error.context_object)
        # Context now includes version info
        self.assertIn("Unitxt", error.context)
        self.assertIn("Python", error.context)
        # Message now includes context box even for "empty" context
        message = str(error)
        self.assertIn("Test error", message)
        self.assertIn("Unitxt Error Context", message)

    def test_error_context_complex_objects(self):
        """Test error_context with complex context objects."""

        class ComplexProcessor:
            def __init__(self, name):
                self.name = name

            def __str__(self):
                return f"ComplexProcessor({self.name})"

        processor = ComplexProcessor("test_processor")

        with self.assertRaises(OSError) as cm:
            with error_context(processor, config_file="settings.json"):
                raise OSError("File not found")

        error = cm.exception
        self.assertEqual(error.context_object, processor)

        # Should identify object by class name, not __str__
        message = str(error)
        self.assertIn("Object: ComplexProcessor", message)
        self.assertIn("Config File: settings.json", message)
