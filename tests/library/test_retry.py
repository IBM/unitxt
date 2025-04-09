import unittest
from unittest.mock import patch

from requests import Timeout as TimeoutError
from requests.exceptions import ConnectionError, HTTPError

# Import your decorator here
from unitxt.utils import retry_connection_with_exponential_backoff


class TestRetryDecorator(unittest.TestCase):
    def setUp(self):
        # Create a mock for time.sleep to avoid waiting during tests
        self.sleep_patcher = patch("time.sleep")
        self.mock_sleep = self.sleep_patcher.start()

        # Create a mock for logging
        self.log_patcher = patch("logging.warning")
        self.mock_log = self.log_patcher.start()

        # Mock settings
        self.settings_patcher = patch("unitxt.settings")
        self.mock_settings = self.settings_patcher.start()
        self.mock_settings.max_retries_for_web_resource = 3

    def tearDown(self):
        self.sleep_patcher.stop()
        self.log_patcher.stop()
        self.settings_patcher.stop()

    def test_successful_function(self):
        """Test that the decorator doesn't interfere with successful function calls."""
        # Function that always succeeds
        @retry_connection_with_exponential_backoff()
        def successful_function():
            return "success"

        result = successful_function()
        self.assertEqual(result, "success")
        self.mock_sleep.assert_not_called()

    def test_direct_retry_exception(self):
        """Test retry with an exception directly in the retry list."""
        # Counter to track number of calls
        counter = {"calls": 0}

        @retry_connection_with_exponential_backoff(max_retries=3)
        def failing_function():
            counter["calls"] += 1
            if counter["calls"] < 3:
                raise ConnectionError("Network error")
            return "success after retry"

        result = failing_function()
        self.assertEqual(result, "success after retry")
        self.assertEqual(counter["calls"], 3)
        self.assertEqual(self.mock_sleep.call_count, 2)

    def test_max_retries_reached(self):
        """Test that it stops retrying after max_retries and raises the exception."""
        @retry_connection_with_exponential_backoff(max_retries=3)
        def always_failing():
            raise TimeoutError("Timeout error")

        with self.assertRaises(TimeoutError):
            always_failing()

        self.assertEqual(self.mock_sleep.call_count, 2)  # Should sleep 2 times for 3 attempts

    def test_chained_exception_with_cause(self):
        """Test retry with exception chained using 'raise from' (__cause__)."""
        counter = {"calls": 0}

        @retry_connection_with_exponential_backoff(max_retries=3)
        def function_with_chained_exception():
            counter["calls"] += 1
            if counter["calls"] < 3:
                try:
                    raise ConnectionError("Original error")
                except ConnectionError as e:
                    new_error = ValueError("Wrapped error")
                    new_error.__cause__ = e
                    raise new_error
            return "success"

        result = function_with_chained_exception()
        self.assertEqual(result, "success")
        self.assertEqual(counter["calls"], 3)
        self.assertEqual(self.mock_sleep.call_count, 2)

    def test_chained_exception_with_context(self):
        """Test retry with exception chained implicitly in except block (__context__)."""
        counter = {"calls": 0}

        @retry_connection_with_exponential_backoff(max_retries=3)
        def function_with_context_exception():
            counter["calls"] += 1
            if counter["calls"] < 3:
                try:
                    raise TimeoutError("Original timeout")
                except Exception:
                    # This creates an implicit __context__ link
                    raise RuntimeError("Another error") from None
            return "success"

        result = function_with_context_exception()
        self.assertEqual(result, "success")
        self.assertEqual(counter["calls"], 3)
        self.assertEqual(self.mock_sleep.call_count, 2)

    def test_deeply_nested_exception(self):
        """Test retry with deeply nested exception chains."""
        counter = {"calls": 0}

        @retry_connection_with_exponential_backoff(max_retries=3)
        def function_with_deeply_nested_exception():
            counter["calls"] += 1
            if counter["calls"] < 3:
                try:
                    # Level 1: HTTPError (should be caught)
                    raise HTTPError("HTTP error")
                except HTTPError as e1:
                    try:
                        # Level 2: wrap with custom exception
                        custom_error = Exception("Custom wrapper")
                        raise custom_error from e1
                    except Exception as e2:
                        try:
                            # Level 3: wrap with OSError (similar to your example)
                            os_error = OSError("Final error")
                            raise os_error from e2
                        except OSError as e3:
                            # Level 4: wrap with RuntimeError
                            runtime_error = RuntimeError("Very nested error")
                            raise runtime_error from e3
            return "success"

        result = function_with_deeply_nested_exception()
        self.assertEqual(result, "success")
        self.assertEqual(counter["calls"], 3)
        self.assertEqual(self.mock_sleep.call_count, 2)

    def test_non_retry_exception(self):
        """Test that non-retry exceptions are re-raised immediately."""
        @retry_connection_with_exponential_backoff(max_retries=3)
        def function_with_non_retry_exception():
            raise ValueError("Not a retry exception")

        with self.assertRaises(ValueError):
            function_with_non_retry_exception()

        self.mock_sleep.assert_not_called()

    def test_mixed_exception_chain(self):
        """Test complex chains with both retry and non-retry exceptions."""
        counter = {"calls": 0}

        @retry_connection_with_exponential_backoff(max_retries=3)
        def function_with_mixed_chain():
            counter["calls"] += 1

            # Only retry for first 2 calls
            if counter["calls"] == 1:
                # Chain with retry exception
                try:
                    raise ValueError("Not a retry exception")
                except ValueError:
                    error = ConnectionError("Network error")
                    raise error from None
            elif counter["calls"] == 2:
                # Chain with non-retry exception last
                try:
                    raise ConnectionError("Network error")
                except ConnectionError as e1:
                    error = RuntimeError("Runtime error")
                    raise error from e1
            else:
                # No retry exception anywhere in the chain
                try:
                    raise ValueError("Not a retry exception")
                except ValueError as e1:
                    error = RuntimeError("Runtime error")
                    raise error from e1

        # Should retry the first call (with ConnectionError),
        # retry the second call (with RuntimeError caused by ConnectionError),
        # and then fail on the third call (only ValueError and RuntimeError)
        with self.assertRaises(RuntimeError):
            function_with_mixed_chain()

        self.assertEqual(counter["calls"], 3)
        self.assertEqual(self.mock_sleep.call_count, 2)

    def test_custom_backoff_factor(self):
        """Test that custom backoff factor is used."""
        counter = {"calls": 0}

        @retry_connection_with_exponential_backoff(max_retries=3, backoff_factor=2)
        def failing_function():
            counter["calls"] += 1
            if counter["calls"] < 3:
                raise ConnectionError("Network error")
            return "success"

        # Mock random.uniform to return 0.5 for predictable results
        with patch("random.uniform", return_value=0.5):
            failing_function()

            # First wait should be 2*(2^0) + 0.5 = 2.5
            # Second wait should be 2*(2^1) + 0.5 = 4.5
            self.mock_sleep.assert_any_call(2.5)
            self.mock_sleep.assert_any_call(4.5)

    def test_custom_exceptions(self):
        """Test with custom exception types."""
        # Define custom exceptions
        class CustomErrorOne(Exception):
            """Custom exception for testing."""

        class CustomErrorTwo(Exception):
            """Custom exception for testing."""

        counter = {"calls": 0}

        @retry_connection_with_exponential_backoff(
            max_retries=3,
            retry_exceptions=(CustomErrorOne, CustomErrorTwo)
        )
        def function_with_custom_exception():
            counter["calls"] += 1
            if counter["calls"] < 3:
                raise CustomErrorOne("Custom error")
            return "success"

        result = function_with_custom_exception()
        self.assertEqual(result, "success")
        self.assertEqual(counter["calls"], 3)
        self.assertEqual(self.mock_sleep.call_count, 2)

    def test_huggingface_like_error_chain(self):
        """Test that simulates the HuggingFace error chain from your example."""
        counter = {"calls": 0}

        class HfHubHTTPErrorException(Exception):
            """HuggingFace Hub HTTP error for testing."""

        class LocalEntryNotFoundErrorException(Exception):
            """HuggingFace local entry not found error for testing."""

        @retry_connection_with_exponential_backoff(max_retries=3)
        def simulate_huggingface_error():
            counter["calls"] += 1
            if counter["calls"] < 3:
                # Create an error chain similar to the HuggingFace example
                try:
                    # Step 1: HTTP Error (retryable)
                    error1 = HTTPError("429 Client Error: Too Many Requests")
                    raise error1
                except HTTPError as e1:
                    try:
                        # Step 2: HuggingFace HTTP error
                        error2 = HfHubHTTPErrorException("HF Hub HTTP Error")
                        raise error2 from e1
                    except HfHubHTTPErrorException as e2:
                        try:
                            # Step 3: LocalEntryNotFoundError
                            error3 = LocalEntryNotFoundErrorException("Entry not found")
                            raise error3 from e2
                        except LocalEntryNotFoundErrorException as e3:
                            # Step 4: OSError (final error raised to user)
                            error4 = OSError("We couldn't connect to 'https://huggingface.co'")
                            raise error4 from e3
            return "success"

        result = simulate_huggingface_error()
        self.assertEqual(result, "success")
        self.assertEqual(counter["calls"], 3)
        self.assertEqual(self.mock_sleep.call_count, 2)


if __name__ == "__main__":
    unittest.main()
