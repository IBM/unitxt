import argparse
import json
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

# Check for valid net_time in the main performance profile
if main_perf["net_time"] == 0:
    print("Net run time on main is 0, can't calculate ratio of times.")
    sys.exit(1)

# Calculate the ratio between PR and main branch net times
ratio = pr_perf["net_time"] / main_perf["net_time"]

# Markdown table formatting
table_header = "| Branch       | Tot Time (sec) | Load Time (sec) | Net Time (sec) | Performance Ratio |\n"
table_divider = "|--------------|----------------|-----------------|----------------|-------------------|\n"
table_main = f"| Main Branch  | {main_perf['total_time']:<14} | {main_perf['load_time']:<14} | {main_perf['net_time']:<14} | -                 |\n"
table_pr = f"| PR Branch    | {pr_perf['total_time']:<14} | {pr_perf['load_time']:<14} | {pr_perf['net_time']:<14} | {ratio:.2f}            |\n"

# Print markdown table
print("### Performance Comparison Results\n")
print(table_header + table_divider + table_main + table_pr)
print("\n\nParticipating cards: ", pr_perf["cards_tested"])
print("Main branch use_eager_execution: ", main_perf["used_eager_mode"])
print("PR branch use_eager_execution: ", pr_perf["used_eager_mode"])

# Performance degradation check (5% threshold)
if ratio > 1.05:
    print("\n**Warning**: Performance degradation exceeds 5%!")
    print(
        "Explore branch performance via 'python performance_profile/card_profiler.py --output_file=<path to json file>',"
        "followed by 'snakeviz <the performance.prof file specified in the output json file>'."
    )
    sys.exit(1)

print("\nPerformance of the PR branch is within acceptable limits.")
