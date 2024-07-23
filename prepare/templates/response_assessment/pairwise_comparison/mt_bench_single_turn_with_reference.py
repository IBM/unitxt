from unitxt import add_to_catalog
from unitxt.templates import PairwiseChoiceTemplate

for to_shuffle in [True, False]:
    add_to_catalog(
        PairwiseChoiceTemplate(
            choice_a_field="answer_a",
            choice_b_field="answer_b",
            answer_field="winner",
            choice_a_label="A",
            choice_b_label="B",
            choice_tie_label="C",
            shuffle=to_shuffle,
            instruction="Please act as an impartial judge and evaluate the quality of the responses provided by two AI"
            " assistants to the user question displayed below. Your evaluation should consider correctness"
            " and helpfulness. You will be given a reference answer, assistant A's answer, and assistant"
            " B's answer. Your job is to evaluate which assistant's answer is better. Begin your evaluation"
            " by comparing both assistants' answers with the reference answer. Identify and correct any"
            " mistakes. Avoid any position biases and ensure that the order in which the responses were"
            " presented does not influence your decision. Do not allow the length of the responses to"
            " influence your evaluation. Do not favor certain names of the assistants. Be as objective"
            " as possible. After providing your explanation, output your final verdict by strictly"
            ' following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better,'
            ' and "[[C]]" for a tie.\n\n',
            input_format="[User Question]\n{question}\n\n"
            "[The Start of Reference Answer]\n{reference_answer}\n[The End of Reference Answer]\n\n"
            "[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n"
            "[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]",
            output_format="[[{winner}]]",
            postprocessors=[
                r"processors.extract_mt_bench_label_judgment",
            ],
        ),
        f"templates.response_assessment.pairwise_comparison.mt_bench_single_turn_with_reference{'_with_shuffling' if to_shuffle else ''}",
        overwrite=True,
    )
