import argparse
import json
import os
import sys

# Argument parser to get file paths from the command line
parser = argparse.ArgumentParser(description="Compare performance profiles.")
parser.add_argument(
    "main_perf_file", type=str, help="Path to main performance profile JSON file"
)
parser.add_argument(
    "pr_perf_file", type=str, help="Path to PR performance profile JSON file"
)
args = parser.parse_args()

# Reading both performance JSON files:
with open(args.main_perf_file) as openfile:
    main_perf = json.load(openfile)

with open(args.pr_perf_file) as openfile:
    pr_perf = json.load(openfile)

print(f"dataset_query = \"{main_perf['dataset_query']}\"")
print(f"used_eager_mode in main = {main_perf['used_eager_mode']}")
print(f"used_eager_mode in PR = {pr_perf['used_eager_mode']}")
print(f"use Mocked inference = {os.environ['UNITXT_MOCK_INFERENCE_MODE']}")

ratio1 = (
    (pr_perf["generate_benchmark_dataset_time"] - pr_perf["load_time_no_initial_ms"])
    / (
        main_perf["generate_benchmark_dataset_time"]
        - main_perf["load_time_no_initial_ms"]
    )
    if (
        main_perf["generate_benchmark_dataset_time"]
        - main_perf["load_time_no_initial_ms"]
    )
    > 0
    else 1
)
ratio2 = (
    pr_perf["evaluation_time"] / main_perf["evaluation_time"]
    if main_perf["evaluation_time"] > 0
    else 1
)
# Markdown table formatting

line1 = "  What is Measured  | Main Branch |  PR Branch  | PR/Main ratio \n"
line2 = "--------------------|-------------|-------------|---------------\n"
line3 = f" Total time         | {main_perf['total_time']:>11} | {pr_perf['total_time']:>11} | {pr_perf['total_time']/main_perf['total_time']:.2f}\n"
line4 = f" Load time          | {main_perf['load_time_no_initial_ms']:>11} | {pr_perf['load_time_no_initial_ms']:>11} | {pr_perf['load_time_no_initial_ms']/main_perf['load_time_no_initial_ms']:.2f}\n"
line5 = f" DS Gen. inc. Load  | {main_perf['generate_benchmark_dataset_time']:>11} | {pr_perf['generate_benchmark_dataset_time']:>11} | {pr_perf['generate_benchmark_dataset_time']/main_perf['generate_benchmark_dataset_time']:.2f}\n"
line6 = f" DS Gen. exc. Load  | {round(main_perf['generate_benchmark_dataset_time']-main_perf['load_time_no_initial_ms'], 3):>11} | {round(pr_perf['generate_benchmark_dataset_time']-pr_perf['load_time_no_initial_ms'], 3):>11} | {ratio1:.2f}\n"
line7 = f" Inference time     | {main_perf['inference_time']:>11} | {pr_perf['inference_time']:>11} | {pr_perf['inference_time']/main_perf['inference_time']:.2f}\n"
line8 = f" Evaluate  time     | {main_perf['evaluation_time']:>11} | {pr_perf['evaluation_time']:>11} | {ratio2:.2f}\n"
line9 = f" Benchmark Instant. | {main_perf['instantiate_benchmark_time']:>11} | {pr_perf['instantiate_benchmark_time']:>11} | {pr_perf['instantiate_benchmark_time']/main_perf['instantiate_benchmark_time']:.2f}\n"
line10 = f" Model Instantiation| {main_perf['instantiate_model_time']:>11} | {pr_perf['instantiate_model_time']:>11} | {pr_perf['instantiate_model_time']/main_perf['instantiate_model_time']:.2f}\n"

print("### Performance Comparison Results, time expressed in seconds:\n")
print(line1 + line2 + line3 + line4 + line5 + line6 + line7 + line8 + line9 + line10)
print("\n\n")
# Performance degradation check (5% threshold)
if ratio1 > 1.05 or ratio2 > 1.05:
    print(
        "\n**Warning**: Performance degradation in Dataset Generation and/or Evaluation exceeds 5%!"
    )
    print(
        "Explore branch performance via 'python performance/bluebench_profiler.py --output_file=<path to json file>',"
        "followed by 'snakeviz <the performance.prof file specified in the output json file>'."
    )
    sys.exit(1)

print("\nPerformance of the PR branch is within acceptable limits.")
