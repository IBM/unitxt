import csv

from ..logging_utils import get_logger
from .run import run_unitxt
from .ui_utils import data

logger = get_logger()

output_file = "ui_tester.csv"
headers = ["task", "dataset", "template", "num_shots", "prompt", "is_failed"]

if __name__ == "__main__":
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
                    logger.info(row)
                    with open(output_file, "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(row)
