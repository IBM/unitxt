import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import List

import requests
from unitxt import add_to_catalog
from unitxt.card import TaskCard
from unitxt.loaders import LoadFromDictionary
from unitxt.task import Task
from unitxt.templates import MultipleChoiceTemplate
from unitxt.templates import TemplatesList, InputOutputTemplate
from unitxt.test_utils.card import test_card

os.environ["UNITXT_ALLOW_UNVERIFIED_CODE"] = "True"
dataset_name = "social_iqa"


def format_to_unitxt(jsonl_path, labels_path):
    # Read JSONL file
    with open(jsonl_path, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]

    # Read labels file
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f]

    formatted_data = []

    for example, label in zip(examples, labels):
        # Get the correct answer text based on the label
        label_map = {"1": 0, "2": 1, "3": 2}
        choices = [example["answerA"], example["answerB"], example["answerC"]]
        options = ["A", "B", "C"]
        formatted_data.append(
            {
                "context": example["context"],
                "question": example["question"],
                "choices": choices,
                "options": options,
                "answer": label_map[label],
                "label": choices[label_map[label]],
            }
        )

    return formatted_data


def process_zip_content(url):
    print("Downloading and processing Social IQa dataset...")
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Download and process in memory
        with requests.get(url, stream=True) as response:
            response.raise_for_status()

            # Create a bytes buffer from the streamed content
            zip_buffer = io.BytesIO()
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    zip_buffer.write(chunk)

            zip_buffer.seek(0)

            # First, let's examine the zip contents
            with zipfile.ZipFile(zip_buffer) as zip_ref:
                # Print all files in the zip
                print("Files in zip archive:", zip_ref.namelist())

                # Extract to temporary directory
                zip_ref.extractall(temp_dir_path)

                # List all files in temp directory to verify extraction
                print("Files in temp directory:", os.listdir(temp_dir_path))

                # Find the correct paths by searching recursively
                train_jsonl_file = None
                train_labels_file = None

                for root, dirs, files in os.walk(temp_dir_path):
                    for file in files:
                        if file == "train.jsonl":
                            train_jsonl_file = Path(root) / file
                        elif file == "train-labels.lst":
                            train_labels_file = Path(root) / file
                        elif file == "dev.jsonl":
                            dev_jsonl_file = Path(root) / file
                        elif file == "dev-labels.lst":
                            dev_labels_file = Path(root) / file
                        # Process dev data
                print(f"Found JSONL file at: {train_jsonl_file}")
                print(f"Found labels file at: {train_labels_file}")

                if train_jsonl_file is None or train_labels_file is None:
                    raise FileNotFoundError(
                        "Could not find required files in the zip archive"
                    )

                # Process the files
                train_data = format_to_unitxt(train_jsonl_file, train_labels_file)

                if dev_jsonl_file is None or dev_labels_file is None:
                    raise FileNotFoundError(
                        "Could not find required files in the zip archive"
                    )

                dev_data = format_to_unitxt(dev_jsonl_file, dev_labels_file)

                return train_data, dev_data


def create_card(data):
    loader = LoadFromDictionary(data=data)

    task = Task(
        input_fields={
            "context": str,
            "question": str,
            "choices": List[str],
            "options": List[str],
        },
        reference_fields={
            "answer": int,
            "choices": List[str],
            "options": List[str],
        },
        prediction_type=str,
        metrics=["metrics.accuracy"],
    )

    template = MultipleChoiceTemplate(
        input_format="Context: {context}\nQuestion: {question}\nOptions:\n{choices}\nWhich is the most appropriate answer?",
        postprocessors=["processors.first_character"],
        target_field="answer",
    )
    templates = TemplatesList(
        [
            template,
            InputOutputTemplate(
                input_format="Context: {context}\nQuestion: {question}\nOptions:\n{choices}\nWhich is the most appropriate answer?",
                output_format="{answer}",
            ),
        ],
    )

    card = TaskCard(
        loader=loader,
        task=task,
        templates=templates,
    )

    test_card(card, debug=True)
    add_to_catalog(card, f"cards.{dataset_name}.multiple_choice", overwrite=True)


def main():
    # URL of the dataset
    url = "https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip"
    train_data, dev_data = process_zip_content(url)
    # Create the final dictionary

    # Create the final dictionary
    formatted_dataset = {
        "train": train_data,
        "validation": dev_data,  # Using dev as test for now
    }
    create_card(formatted_dataset)


if __name__ == "__main__":
    main()
