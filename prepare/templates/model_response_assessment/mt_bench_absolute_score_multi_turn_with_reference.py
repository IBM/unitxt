from unitxt.catalog import add_to_catalog
from unitxt.templates import ChatTemplate, DialogFieldsData

add_to_catalog(
    ChatTemplate(
        dialog_fields=[
            DialogFieldsData(
                dialog_field="dialog",
                assistant_role_label="### ### Reference answer:",
                user_role_label="### User:",
                system_role_label="### System:",
            ),
            DialogFieldsData(
                dialog_field="reference_dialog",
                assistant_role_label="### Assistant A:",
                user_role_label="### User:",
                system_role_label="### System:",
            ),
        ],
        turns_separator="\n\n",
        label_separator="\n",
        instruction="Please act as an impartial judge and evaluate the quality of the response provided by an AI"
        " assistant to the user question. Your evaluation should consider correctness and helpfulness."
        " You will be given a reference answer and the assistant's answer. You evaluation should focus"
        " on the assistant's answer to the second question. Begin your evaluation by comparing the"
        " assistant's answer with the reference answer. Identify and correct any mistakes."
        " Be as objective as possible. After providing your explanation, you must rate the response on"
        ' a scale of 1 to 10 by strictly following this format: "[[rating]]",'
        ' for example: "Rating: [[5]]".\n\n',
        input_format="<|The Start of Reference Answer|>\n\n"
        "{reference_dialog}\n\n"
        "<|The End of Reference Answer|>\n\n\n"
        "<|The Start of Assistant A's Conversation with User|>\n\n"
        "{dialog}\n\n"
        "<|The End of Assistant A's Conversation with User|>",
        output_format="[[{rating_label}]]",
        postprocessors=[
            r"processors.extract_mt_bench_judgment",
        ],
    ),
    "templates.model_response_assessment.mt_bench_absolute_score_single_turn_with_reference",
    overwrite=True,
)
