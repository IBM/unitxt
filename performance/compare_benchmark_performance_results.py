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

print(f'dataset_query = "{main_perf["dataset_query"]}"')
print(f"used_eager_mode in main = {main_perf['used_eager_mode']}")
print(f"used_eager_mode in PR = {pr_perf['used_eager_mode']}")
print(f"use Mocked inference = {os.environ['UNITXT_MOCK_INFERENCE_MODE']}")
print("Raw datasets, that are loaded and processed here, are assumed to reside in local file ststem when the run starts.")

ratio0 = pr_perf["total_time"] / main_perf["total_time"]

gen_via_recipe_pr = round(pr_perf["instantiate_benchmark_time"] + pr_perf["generate_benchmark_dataset_time"], 3)
gen_via_recipe_main = round(main_perf["instantiate_benchmark_time"] + main_perf["generate_benchmark_dataset_time"], 3)
ratio1 = gen_via_recipe_pr / gen_via_recipe_main if gen_via_recipe_main > 0 else 1
ratio2 = (
    pr_perf["evaluation_time"] / main_perf["evaluation_time"]
    if main_perf["evaluation_time"] > 0
    else 1
)
# Markdown table formatting

line1 = "  What is Measured  | Main Branch |  PR Branch  | PR/Main ratio \n"
line2 = "--------------------|-------------|-------------|---------------\n"
line3 = f" Unitxt load_dataset| {main_perf['total_time']:>11} | {pr_perf['total_time']:>11} | {ratio0:.2f}\n"
line4 = f" DS Gen via recipe  | {gen_via_recipe_main:>11} | {gen_via_recipe_pr:>11} | {ratio1:.2f}\n"
line5 = f" Model Instantiation| {main_perf['instantiate_model_time']:>11} | {pr_perf['instantiate_model_time']:>11} | {pr_perf['instantiate_model_time'] / main_perf['instantiate_model_time']:.2f}\n"
line6 = f" Inference time     | {main_perf['inference_time']:>11} | {pr_perf['inference_time']:>11} | {pr_perf['inference_time'] / main_perf['inference_time']:.2f}\n"
line7 = f" Evaluate  time     | {main_perf['evaluation_time']:>11} | {pr_perf['evaluation_time']:>11} | {ratio2:.2f}\n"

print("### Performance Comparison Results, time expressed in seconds:\n")
print(line1 + line2 + line3 + line4 + line5 + line6 + line7)
print("\n\n")
# Performance degradation check (15% threshold)
if (ratio0 > 1.15) and (ratio1 > 1.15 or ratio2 > 1.15):
    print(
        "\n**Warning**: Performance degradation in Dataset Generation and/or Evaluation exceeds 15%!"
    )
    print(
        "Explore branch performance via 'python performance/bluebench_profiler.py --output_file=``path to json file``',"
        "followed by 'snakeviz ``the performance.prof file specified in the output json file``'."
    )
    sys.exit(1)

print("\nPerformance of the PR branch is within acceptable limits.")
