import ast
import json
import os
import re
import tempfile
from typing import Dict

import pandas as pd
from datasets import DatasetDict
from datasets import load_dataset as hf_load_dataset
from unitxt import add_to_catalog
from unitxt.blocks import (
    FormTask,
    InputOutputTemplate,
    MapInstanceValues,
    TaskCard,
    TemplatesList,
)
from unitxt.dataclass import FinalField
from unitxt.loaders import Loader
from unitxt.operators import DownloadOperator, ZipExtractorOperator
from unitxt.stream import MultiStream

contract_nli_labels_dict = {
    "nda-11": "no reverse engineering",
    "nda-16": "return of confidential information",
    "nda-15": "no licensing",
    "nda-10": "confidentiality of agreement",
    "nda-2": "none-inclusion of non-technical information",
    "nda-1": "explicit identification",
    "nda-19": "survival of obligations",
    "nda-12": "permissible development of similar information",
    "nda-20": "permissible post-agreement possession",
    "nda-3": "inclusion of verbally conveyed information",
    "nda-18": "no solicitation",
    "nda-7": "sharing with third-parties",
    "nda-17": "permissible copy",
    "nda-8": "notice on compelled disclosure",
    "nda-13": "permissible acquirement of similar information",
    "nda-5": "sharing with employees",
    "nda-4": "limited use",
}

dataset_name = "contract_nli"


def download_and_extract(source, target_dir):
    zip_file = os.path.join(tempfile.gettempdir(), "contract-nli.zip")
    DownloadOperator(source, zip_file)()
    ZipExtractorOperator(zip_file, target_dir)()


class ContractNliLoader(Loader):
    files: Dict[str, str] = FinalField(default_factory=dict)

    def process(self):
        tempdir = tempfile.gettempdir()
        dir_with_unzipped_files = os.path.join(tempdir, "contract-nli-raw")
        dir_with_csv_files = os.path.join(tempdir, "contract-nli")
        download_and_extract(
            source="https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip",
            target_dir=dir_with_unzipped_files,
        )

        self.files = {
            split: os.path.join(dir_with_csv_files, f"{split}.csv")
            for split in ["train", "dev", "test"]
        }

        def clean_text(t):
            t = (
                t.replace("\n", " ")
                .replace("  ", " ")
                .replace("    ", " ")
                .replace("&lt;", "<")
                .replace("“", '"')
                .replace("”", '"')
            )
            return re.sub(r"\s+", " ", t)

        ds_dict = DatasetDict()
        os.makedirs(dir_with_csv_files, exist_ok=True)

        for data_split in ["train", "dev", "test"]:
            with open(
                os.path.join(
                    os.path.join(dir_with_unzipped_files, "contract-nli"),
                    f"{data_split}.json",
                )
            ) as in_file:
                data = json.load(in_file)
            examples_for_set = {}
            for document in data["documents"]:
                annotations = document["annotation_sets"][0]["annotations"]
                text = document["text"]
                span_texts = [text[span[0] : span[1]] for span in document["spans"]]
                doc_examples = [
                    (span_texts[span_loc], ann_key)
                    for ann_key, ann_val in annotations.items()
                    if "Entailment" in ann_val["choice"]
                    for span_loc in ann_val["spans"]
                    if not span_texts[span_loc].endswith(":")
                ]
                span_texts = [st for st in span_texts if not st.endswith(":")]
                text_to_classes = {x: [] for x in span_texts}
                for text, label in doc_examples:
                    text_to_classes[text].append(label)
                examples_for_set = {**examples_for_set, **text_to_classes}
            examples_for_set_as_list = list(examples_for_set.items())
            examples_for_set_as_list = [
                (clean_text(x[0]), x[1]) for x in examples_for_set_as_list
            ]
            examples_for_set_as_list = [
                x
                for x in examples_for_set_as_list
                if len(x[1]) > 0 and len(x[0].split()) > 4
            ]
            split = data_split
            out_file = os.path.join(dir_with_csv_files, f"{split}.csv")
            ds_dict[split] = pd.DataFrame()
            ds_dict[split]["labels"] = [x[1] for x in examples_for_set_as_list]
            ds_dict[split]["text"] = [x[0] for x in examples_for_set_as_list]
            ds_dict[split].to_csv(out_file, encoding="utf-8", index=False)

        dataset = hf_load_dataset(
            dir_with_csv_files,
            data_files={
                "train": "train.csv",
                "validation": "dev.csv",
                "test": "test.csv",
            },
            streaming=False,
        )  # TODO labels read as list
        for split in dataset.keys():
            dataset[split] = dataset[split].map(
                lambda example: {
                    "labels": ast.literal_eval(example["labels"]),
                    "text": example["text"],
                }
            )

        return MultiStream.from_iterables(dataset)


card = TaskCard(
    loader=ContractNliLoader(),
    preprocess_steps=[
        MapInstanceValues(
            mappers={"labels": contract_nli_labels_dict}, process_every_value=True
        ),
    ],
    task=FormTask(
        inputs=["text"],
        outputs=["labels"],
        metrics=["metrics.f1_micro", "metrics.accuracy", "metrics.f1_macro"],
    ),
    templates=TemplatesList(
        [
            InputOutputTemplate(
                input_format="{text}",
                output_format="{labels}",
            ),
        ]
    ),
)

add_to_catalog(artifact=card, name=f"cards.{dataset_name}", overwrite=True)

ds = hf_load_dataset("unitxt/data", f"card=cards.{dataset_name},template_card_index=0")
# print(ds['dev']['additional_inputs'][0])
