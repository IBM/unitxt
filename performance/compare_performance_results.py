import argparse
import json
import sys

from unitxt.logging_utils import get_logger

logger = get_logger()

# Argument parser to get file paths from the command line
parser = argparse.ArgumentParser(description="Compare performance profiles.")
parser.add_argument(
    "main_perf_file", type=str, help="Path to main performance profile JSON file"
)
parser.add_argument(
    "pr_perf_file", type=str, help="Path to PR performance profile JSON file"
)
args = parser.parse_args()

# Reading both performance json files:
with open(args.main_perf_file) as openfile:
    main_perf = json.load(openfile)

with open(args.pr_perf_file) as openfile:
    pr_perf = json.load(openfile)

logger.critical(
    f"Net time (total minus load) of running benchmark on main is {main_perf['net_time']}"
)
logger.critical(
    f"Net time (total minus load) of running benchmark on PR branch is {pr_perf['net_time']}"
)

if main_perf["net_time"] == 0:
    logger.critical("Net run time on main is 0, can't calculate ratio of times.")
    sys.exit(1)

ratio = pr_perf["net_time"] / main_perf["net_time"]

if ratio > 1.1:
    logger.critical("Performance degradation exceeds 10% !")
    logger.critical(
        "Explore branch performance via 'python performance_profile/card_profiler.py', followed by 'snakeviz performance_profile/logs/cards_benchmark.prof'"
    )
    sys.exit(1)

logger.critical(
    "Compared to main branch, performance of the PR branch is within acceptable limits."
)
