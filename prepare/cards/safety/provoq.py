from unitxt import add_to_catalog
from unitxt.blocks import InputOutputTemplate, LoadHF, Task, TaskCard, TemplatesDict
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="IBM/ProvoQ", data_classification_policy=["public"]),
    task=Task(
        input_fields={"question": str, "stigma": str, "group": str},
        reference_fields={},
        prediction_type=str,
        metrics=[
            "metrics.llm_as_judge.safety.llama_3_70b_instruct_ibm_watsonx_template_harm_rating"
        ],
    ),
    templates=TemplatesDict(
        {"default": InputOutputTemplate(input_format="{question}", output_format="")}
    ),
    __description__="The ProvoQ dataset is designed to evaluate the sensitivity of large language models (LLMs) to stigma-related topics. It contains a comprehensive set of human-curated provocative questions that systematically target minority-stigma pairs in the United States, creating a diverse and nuanced set of questions that reflect these sensitive topics. The dataset aims to support research in understanding and mitigating biases in AI systems, particularly in the context of minority groups. While most questions are toxic, others may seem benign but potentially elicit harmful responses. The dataset contains questions in text format, organized by minority-stigma pairs.",
    __tags__={
        "languages": ["english"],
    },
)

test_card(
    card,
    strict=False,
    demos_taken_from="test",
    num_demos=0,
)

add_to_catalog(card, "cards.safety.provoq", overwrite=True)
