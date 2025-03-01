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
print("Given the raw datasets stored in the local file system, their processing through the Unitxt pipeline lasts as detailed below.")

ratios = {}
for k in pr_perf:
    if not isinstance(pr_perf[k], float):
        continue
    ratios[k] = pr_perf[k] / main_perf[k] if main_perf[k] > 0 else 1


# Markdown table formatting

line1 = "  What is Measured  | Main Branch |  PR Branch  | PR/Main ratio \n"
line2 = "--------------------|------------:|------------:|--------------:\n"
line3 = f" Unitxt load_recipe | {main_perf['load_recipe']} | {pr_perf['load_recipe']} | {ratios['load_recipe']:.2f}\n"
line4 = f" Source_to_dataset  | {main_perf['source_to_dataset']} | {pr_perf['source_to_dataset']} | {ratios['source_to_dataset']:.2f}\n"
line5 = f" Just load and list | {main_perf['just_load_and_list']} | {pr_perf['just_load_and_list']} | {ratios['just_load_and_list']:.2f}\n"
line6 = f" Stream thru recipe | {main_perf['just_stream_through_recipe']} | {pr_perf['just_stream_through_recipe']} | {ratios['just_stream_through_recipe']:.2f}\n"
line7 = f" Inference time     | {main_perf['inference_time']} | {pr_perf['inference_time']} | {ratios['inference_time']:.2f}\n"
line8 = f" Evaluate  time     | {main_perf['evaluation_time']} | {pr_perf['evaluation_time']} | {ratios['evaluation_time']:.2f}\n"

print("### Performance Comparison Results, time expressed in seconds:\n")
print(line1 + line2 + line3 + line4 + line5 + line6 + line7 + line8)
print("\n\n")
# Performance degradation check (15% threshold)
if (ratios["source_to_dataset"] > 1.15) and (ratios["just_stream_through_recipe"] > 1.15 or ratios["evaluation_time"] > 1.15):
    print(
        "\n**Warning**: Performance degradation in Dataset Generation and/or Evaluation exceeds 15%!"
    )
    print(
        "Explore branch performance via 'python performance/bluebench_profiler.py --output_file=``path to json file`` --employ_cProfile',"
        "followed by 'snakeviz ``the performance.prof file specified in the output json file``'."
    )
    sys.exit(1)

print("\nPerformance of the PR branch is within acceptable limits.")
