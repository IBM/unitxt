import json
import os
import sys

import pandas as pd
from unitxt.logging_utils import get_logger

logger = get_logger()


def extract_scores(directory):
    data = []

    for filename in sorted(os.listdir(directory)):
        if filename.endswith("evaluation_results.json"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = json.load(f)

                    env_info = content.get("environment_info", {})
                    timestamp = env_info.get("timestamp_utc", "N/A")
                    model = env_info.get("parsed_arguments", {}).get("model", "N/A")
                    results = content.get("results", {})

                    row = {}
                    row["Model"] = model
                    row["Timestamp"] = timestamp
                    row["Average"] = results.get("score", "N/A")

                    for key in results.keys():
                        if isinstance(results[key], dict):
                            score = results[key].get("score", "N/A")
                            row[key] = score

                    data.append(row)
            except Exception as e:
                logger.error(f"Error parsing results file {filename}: {e}.")

    return pd.DataFrame(data).sort_values(by="Timestamp", ascending=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python summarize_cli_results.py <results-directory>")
        sys.exit(1)

    directory = sys.argv[1]
    df = extract_scores(directory)

    logger.info(df.to_markdown(index=False))
