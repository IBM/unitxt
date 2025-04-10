from unitxt import add_to_catalog
from unitxt.blocks import InputOutputTemplate, Task, TaskCard
from unitxt.loaders import LoadHF, MultipleSourceLoader
from unitxt.operators import SelectFields
from unitxt.stream_operators import JoinStreams
from unitxt.templates import TemplatesDict
from unitxt.test_utils.card import test_card

dataset_path = "stanford-crfm/air-bench-2024"

card = TaskCard(
    loader=MultipleSourceLoader( sources =
    [
        LoadHF(path=dataset_path, name="default"),
        LoadHF(path=dataset_path, data_files={
            "judge_prompts": "judge_prompt_final.csv"
        })
    ],
        data_classification_policy=["public"],
    ),
    preprocess_steps=[
        SelectFields(
            fields=["cate-idx", "judge_prompt"], apply_to_streams=["judge_prompts"]
        ),
        JoinStreams(
            left_stream="test",
            right_stream="judge_prompts",
            how="inner",
            on=["cate-idx"],
            new_stream_name="test",)
    ],
    task=Task(
        input_fields={"cate-idx": str, "l2-name": str, "l3-name": str, "l4-name": str, "prompt": str, "judge_prompt": str},
        reference_fields={},
        prediction_type=str,
        metrics=[
            "metrics.granite_guardian.assistant_risk.harm[prediction_type=str,user_message_field=prompt,assistant_message_field=prediction]",
        ],
    ),
    templates=TemplatesDict({"default": InputOutputTemplate(input_format="{prompt}\n", output_format="")}),
    __description__="AIRBench 2024 is a AI safety benchmark that aligns with emerging government regulations and company policies. It consists of diverse, malicious prompts spanning categories of the regulation-based safety categories in the AIR 2024 safety taxonomy.",
    __tags__={
        "languages": ["english"],
        "urls": {"arxiv": "https://arxiv.org/abs/2407.17436"},
    },
)

test_card(card,strict=False,loader_limit=10000)

add_to_catalog(card, "cards.safety.airbench2024", overwrite=True)
