from unitxt import add_to_catalog
from unitxt.blocks import LoadHF, TaskCard
from unitxt.collections_operators import Dictify
from unitxt.operators import ListFieldValues, Rename, SelectFields, Set
from unitxt.stream_operators import JoinStreams
from unitxt.test_utils.card import test_card

dataset_path = "stanford-crfm/air-bench-2024"
dataset_name = "default"
judge_prompts = "judge_prompts"
context_fields = ["l2-name", "l3-name", "l4-name", "judge_prompt"]

card = TaskCard(
    loader=LoadHF(
        path=dataset_path,
        data_files={
            dataset_name: "category*.csv",
            judge_prompts: "judge_prompt_final.csv",
        },
        data_classification_policy=["public"],
    ),
    preprocess_steps=[
        Rename(field="prompt", to_field="input", apply_to_streams=[dataset_name]),
        Set(
            fields={
                "output": "",
                "type_of_input": "question",
                "type_of_output": "response",
            },
            apply_to_streams=[dataset_name],
        ),
        SelectFields(
            fields=["cate-idx", "judge_prompt"], apply_to_streams=[judge_prompts]
        ),
        JoinStreams(
            left_stream=dataset_name,
            right_stream=judge_prompts,
            how="inner",
            on=["cate-idx"],
            new_stream_name="test",
        ),
        ListFieldValues(fields=context_fields, to_field="contexts"),
        Dictify(field="contexts", with_keys=context_fields, to_field="contexts"),
    ],
    task="tasks.generation.with_contexts",
    templates=["templates.generation.empty"],
    __description__="AIRBench 2024 is a AI safety benchmark that aligns with emerging government regulations and company policies. It consists of diverse, malicious prompts spanning categories of the regulation-based safety categories in the AIR 2024 safety taxonomy.",
    __tags__={
        "languages": ["english"],
        "urls": {"arxiv": "https://arxiv.org/abs/2407.17436"},
    },
)

test_card(card, strict=False)
add_to_catalog(card, "cards.safety.airbench2024", overwrite=True)
