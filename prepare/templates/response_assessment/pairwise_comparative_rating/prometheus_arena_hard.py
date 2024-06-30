from unitxt import add_to_catalog
from unitxt.templates import PairwiseComparativeRatingTemplate

for to_shuffle in [True, False]:
    add_to_catalog(
        PairwiseComparativeRatingTemplate(
            choice_a_field="answer_a",
            choice_b_field="answer_b",
            choice_a_id_field="model_a",
            choice_b_id_field="model_b",
            answer_field="answer_a_preference",
            shuffle=to_shuffle,
            instruction="###Task Description:\n An instruction (might include an Input inside it), a response to evaluate,"
            " and a score rubric representing a evaluation criteria are given.\n"
            "1. Write a detailed feedback that assess the quality of two responses strictly based on the given score rubric, not evaluating in general.\n"
            "2. After writing a feedback, choose a better response between Response A and Response B. You should refer to the score rubric.\n"
            "3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (A or B)\"\n"
            "4. Please do not generate any other opening, closing, and explanations.\n\n" 
            "###Instruction:\n"
            "Please act as an impartial judge and evaluate the quality of the responses provided by two AI\n"
            " assistants to the user prompt displayed below. You will be given assistant A's answer and"
            " assistant B's answer. Your job is to evaluate which assistant's answer is better."
            "\n\nBegin your evaluation by generating your own answer to the prompt. You must provide"
            " your answers before judging any answers.\n\nWhen evaluating the assistants' answers,"
            " compare both assistants' answers with your answer. You must identify and correct any mistakes or"
            " inaccurate information.\n\nThen consider if the assistant's answers are helpful, relevant,"
            " and concise. Helpful means the answer correctly responds to the prompt or follows the"
            " instructions. Note when user prompt has any ambiguity or more than one interpretation,"
            " it is more helpful and appropriate to ask for clarifications or more information from the"
            " user than providing an answer based on assumptions. Relevant means all parts of the response"
            " closely connect or are appropriate to what is being asked. Concise means the response is"
            " clear and not verbose or excessive.\n\nThen consider the creativity and novelty of the"
            " assistant's answers when needed. Finally, identify any missing important information in"
            " the assistants' answers that would be beneficial to include when responding to the user"
            " prompt."
            "\n{question}\n\n"
            "###Response A:\n"
            "{answer_a}\n\n"
            "###Response B:\n"
            "{answer_b}\n\n"
            "###Score Rubric:\n\n"
            "You must output only one of the following choices"
            " as your final verdict with a label:\n\n"
            "1. Assistant A is significantly better: [[A>>B]]\n"
            "2. Assistant A is slightly better: [[A>B]]\n"
            "3. Tie, relatively the same: [[A=B]]\n"
            "4. Assistant B is slightly better: [[B>A]]\n"
            "5. Assistant B is significantly better: [[B>>A]]\n\n"
            'Example output: "My final verdict is tie: [[A=B]]".',
            postprocessors=["processors.extract_arena_hard_numerical_judgment"],
            output_format="{answer_a_preference}",
        ),
        f"templates.response_assessment.pairwise_comparative_rating.prometheus_arena_hard{'_with_shuffling' if to_shuffle else ''}",
        overwrite=True,
    )
