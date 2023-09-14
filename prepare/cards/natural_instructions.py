import os.path

import numpy as np
from datasets import load_dataset
from src.unitxt.blocks import InputOutputTemplate, LoadHF, SplitRandomMix, TemplatesList
from src.unitxt.card import TaskCard
from src.unitxt.catalog import add_to_catalog
from src.unitxt.instructions import InstructionsList, TextualInstruction
from src.unitxt.operators import CopyFields, FilterByValues
from src.unitxt.prepare_utils.card_types import addClassificationChoices
from src.unitxt.task import FormTask
from src.unitxt.test_utils.card import test_card

hf_df = load_dataset("Muennighoff/natural-instructions")
tasks_names = []
for split in ["train"]:
    # for split in ["train", "validation", "test"]:
    names = np.unique(hf_df[split]["task_name"])
    tasks_names.append(names)

    pandas_df_split = hf_df[split].to_pandas()

    for task in names:
        print("task name:", task)
        json_url = f"https://raw.githubusercontent.com/allenai/natural-instructions/master/tasks/{task}.json"
        definitions = pandas_df_split[pandas_df_split["task_name"] == task]["definition"].unique().tolist()
        assert len(definitions) == 1
        print(definitions)
        task_instruction = definitions[0]
        task = task.replace("-", "_")

        if os.path.isfile(
            f"/u/shachardon/repo/unitxt/src/unitxt/catalog/cards/natural_instructions/{split}/{task}.json"
        ):
            print("already exists. skipping")
            continue

        template = InputOutputTemplate(
            input_format=f"##Instruction: {task_instruction}".strip() + """\n\n##Input: {input}\n\n##Output: """,
            output_format="{target}",
        )
        add_to_catalog(template, f"templates.natural_instructions.{split}.{task}", overwrite=True)

        instruction = TextualInstruction(task_instruction)
        add_to_catalog(instruction, f"instructions.natural_instructions.{split}.{task}", overwrite=True)

        card = TaskCard(
            loader=LoadHF("json", data_files=json_url, field="Instances"),
            preprocess_steps=[
                SplitRandomMix({"train": "train[90%]", "validation": "train[5%]", "test": "train[5%]"}),
                CopyFields(field_to_field=[["output/0", "target"]], use_query=True),
            ],
            task=FormTask(inputs=["input"], outputs=["target"], metrics=["metrics.rouge"]),
            instructions=InstructionsList([f"instructions.natural_instructions.{split}.{task}"]),
            templates=TemplatesList([f"templates.natural_instructions.{split}.{task}"]),
        )

        try:
            test_card(card)
        except Exception as e:
            print("error while generating task", task)
            print(e)

        add_to_catalog(card, f"cards.natural_instructions.{split}.{task}", overwrite=True)
