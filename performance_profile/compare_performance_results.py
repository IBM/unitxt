import json
import sys

from unitxt.logging_utils import get_logger

logger = get_logger()

# Reading both performance json files:
with open("performance_profile/logs/main_cards_benchmark.json") as openfile:
    main_perf = json.load(openfile)

with open("performance_profile/logs/pr_cards_benchmark.json") as openfile:
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
    "Compared to main branch, performance or the PR branch is within acceptable limits."
)
