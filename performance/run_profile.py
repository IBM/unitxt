import fnmatch
import importlib.util
import os
import sys
import time

import psutil

# Dictionary to track function statistics
func_stats = {}

# Get the process instance for memory usage tracking
process = psutil.Process(os.getpid())

# Define include and exclude file patterns
include_files = ["**/src/**/*.py"]  # Include all Python files in the src folder
exclude_files = [
    "**/src/unitxt/generator_utils.py",
    "**/src/unitxt/operator.py",
    "**/src/unitxt/dataclass.py",
    "**/src/unitxt/register.py",
    "**/src/unitxt/__init__.py",
]  # Exclude other files

# Set a threshold for total time to display in the result
total_time_threshold = 0.001  # Only show functions with more than 0.01s total time
top_k = 50  # Number of top functions to display


def should_profile(file_name):
    """Determine whether a file should be profiled based on inclusion/exclusion patterns."""
    # If include_files is specified, check if the file matches any of the patterns
    if include_files:
        included = any(fnmatch.fnmatch(file_name, pattern) for pattern in include_files)
        if not included:
            return False

    # If exclude_files is specified, check if the file matches any of the patterns
    if exclude_files:
        excluded = any(fnmatch.fnmatch(file_name, pattern) for pattern in exclude_files)
        if excluded:
            return False

    return True


def profile_func(frame, event, arg):
    func_name = frame.f_code.co_name
    file_name = frame.f_code.co_filename
    line_number = frame.f_code.co_firstlineno

    # Filter functions based on include/exclude file patterns
    if not should_profile(file_name):
        return

    func_key = (func_name, file_name, line_number)

    if event == "call":  # When a function is called
        # Initialize tracking if it's the first time seeing this function
        if func_key not in func_stats:
            func_stats[func_key] = {
                "calls": 0,
                "tottime": 0.0,
                "cumtime": 0.0,
                "total_memory": 0.0,
                "start_time": 0.0,
                "start_memory": 0.0,
                "file_name": file_name,
                "line_number": line_number,
            }

        # Increment the call count
        func_stats[func_key]["calls"] += 1

        # Record start time and memory usage
        frame.f_locals["start_time"] = time.time()
        frame.f_locals["start_memory"] = process.memory_info().rss  # Memory in bytes

    elif event == "return":  # When the function returns
        start_time = frame.f_locals.get("start_time", None)
        start_memory = frame.f_locals.get("start_memory", None)
        if start_time is not None and start_memory is not None:
            # Calculate elapsed time and memory usage
            elapsed_time = time.time() - start_time
            end_memory = process.memory_info().rss
            memory_used = end_memory - start_memory

            # Update stats
            func_stats[func_key]["tottime"] += elapsed_time
            func_stats[func_key]["cumtime"] += elapsed_time
            func_stats[func_key]["total_memory"] += memory_used


# Function to print stats in cProfile-like format with memory in MB
def print_profile_stats():
    header = "   ncalls  tottime  percall  cumtime  totmem  percall  filename:lineno(function)"
    print(header)
    print("=" * len(header))

    # Filter out functions that don't meet the total time threshold
    filtered_stats = {
        key: stats
        for key, stats in func_stats.items()
        if stats["tottime"] >= total_time_threshold
    }

    # Sort functions by total time (tottime) in descending order
    sorted_stats = sorted(
        filtered_stats.items(), key=lambda item: item[1]["tottime"], reverse=True
    )

    # Limit to top K functions
    top_stats = sorted_stats[:top_k]

    for func_key, stats in top_stats:
        func_name, file_name, line_number = func_key
        calls = stats["calls"]
        tottime = stats["tottime"]
        cumtime = stats["cumtime"]
        total_memory = stats["total_memory"] / (1024 * 1024)  # Convert bytes to MB
        percall_time = tottime / calls if calls > 0 else 0
        memory_per_call = total_memory / calls if calls > 0 else 0

        # Format and print the data like cProfile
        print(
            f"{calls:8d}  {tottime:8.3f}  {percall_time:8.3f}  {cumtime:8.3f}  "
            f"{total_memory:8.1f}  {memory_per_call:8.1f}  "
            f"{file_name}:{line_number}({func_name})"
        )


# Function to import and execute a file from the first command-line argument
def import_and_run_module(file_path):
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = module
    spec.loader.exec_module(module)


if __name__ == "__main__":
    # Get the Python file to profile from the first command-line argument
    if len(sys.argv) < 2:
        print("Usage: python profiler.py <path_to_python_file>")
        sys.exit(1)

    file_to_run = sys.argv[1]

    # Set the profile function
    sys.setprofile(profile_func)

    # Run the specified Python file
    import_and_run_module(file_to_run)

    # Disable profiling when done
    sys.setprofile(None)

    # Print profiling results
    print_profile_stats()
