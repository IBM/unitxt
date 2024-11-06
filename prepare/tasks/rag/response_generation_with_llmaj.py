from typing import List, Union

from unitxt import add_to_catalog
from unitxt.blocks import (
    Task,
)

mapping_str = "judge_to_generator_fields_mapping={ground_truths=reference_answers}"
add_to_catalog(
    Task(
        input_fields={
            "contexts": List[str],
            "contexts_ids": Union[List[int], List[str]],
            "question": str,
        },
        reference_fields={"reference_answers": List[str]},
        metrics=[
            "metrics.rag.response_generation.correctness.token_overlap",
            "metrics.rag.response_generation.faithfullness.token_overlap",
            "metrics.rag.response_generation.correctness.bert_score.deberta_large_mnli",
            f"metrics.llm_as_judge.binary.llama_3_1_70b_instruct_wml_answer_correctness_q_a_gt_loose_logprobs[{mapping_str}]",
            f"metrics.llm_as_judge.binary.llama_3_1_70b_instruct_wml_faithfulness_q_c_a_logprobs[{mapping_str}]",
            f"metrics.llm_as_judge.binary.llama_3_1_70b_instruct_wml_answer_relevance_q_a_logprobs[{mapping_str}]",
            f"metrics.llm_as_judge.binary.llama_3_1_70b_instruct_wml_correctness_holistic_q_c_a_logprobs[{mapping_str}]",
        ],
        augmentable_inputs=["contexts", "question"],
    ),
    "tasks.rag.response_generation_with_wml_llmaj",
    overwrite=True,
)
