import csv

import pandas as pd

from src.unitxt.logging_utils import get_logger
from unitxt.ui.run import run_unitxt
from unitxt.ui.ui_utils import data

logger = get_logger()

output_file = "ui_tester.csv"
headers = ["task", "dataset", "template", "num_shots", "prompt", "is_failed"]

with open(output_file, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers)

run_results = []
for task in data:
    datasets = data[task]
    if "augmentable_inputs" in datasets:
        datasets.pop("augmentable_inputs")
    for dataset in datasets:
        templates = datasets[dataset]
        for template in templates:
            for num_shots in [0, 5]:
                prompt_text, target, pred, result, aggresult, command = run_unitxt(
                    dataset=dataset, template=template, num_demos=num_shots
                )
                is_failed = "Exception:" in prompt_text
                row = [task, dataset, template, num_shots, prompt_text, is_failed]
                with open(output_file, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(row)
                df = pd.read_csv(output_file)
                errors = df[df["is_failed"]].copy()
                pivot = pd.pivot_table(errors, index="prompt", aggfunc="count")
                pivot.sort_values("dataset", ascending=False, inplace=True)
                num_errors = len(errors)
                num_error_types = len(pivot)
                most_common_error = pivot.index.tolist()[0]
                most_common_error_freq = pivot["dataset"].tolist()[0]
                logger.info(
                    f"""
                            {num_errors} errors of {num_error_types} types found so far
                            most common error repeats {most_common_error_freq} times
                            most common error:
                            {most_common_error}
                            """
                )
