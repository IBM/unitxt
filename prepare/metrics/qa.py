from unitxt import add_to_catalog
from unitxt.metrics import MetricsList

add_to_catalog(
    MetricsList(["metrics.rouge"]), "metrics.qa.open.recommended_no_gpu", overwrite=True
)
add_to_catalog(
    MetricsList(["metrics.sentence_bert.bge_large_en_1_5"]),
    "metrics.qa.open.recommended_local_gpu",
    overwrite=True,
)
add_to_catalog(
    MetricsList(
        ["metrics.llm_as_judge.rating.llama_3_70b_instruct.generic_single_turn"]
    ),
    "metrics.qa.open.recommended_llm_as_judge",
    overwrite=True,
)
