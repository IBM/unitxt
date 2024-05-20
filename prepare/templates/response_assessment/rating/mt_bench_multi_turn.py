from unitxt.catalog import add_to_catalog
from unitxt.templates import DialogFieldsData, DialogTemplate

add_to_catalog(
    DialogTemplate(
        dialog_fields=[
            DialogFieldsData(
                dialog_field="dialog",
                assistant_role_label="### Assistant A:",
                user_role_label="### User:",
                system_role_label="### System:",
            ),
        ],
        turns_separator="\n\n",
        label_separator="\n",
        instruction="Please act as an impartial judge and evaluate the quality of the response provided by an"
        " AI assistant to the user question displayed below. Your evaluation should consider factors"
        " such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail"
        " of the response. You evaluation should focus on the assistant's answer to the second user"
        " question. Begin your evaluation by providing a short explanation. Be as objective as possible."
        " After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly"
        ' following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n',
        input_format="<|The Start of Assistant A's Conversation with User|>\n\n"
        "{dialog}\n\n"
        "<|The End of Assistant A's Conversation with User|>",
        output_format="[[{rating}]]",
        postprocessors=[
            r"processors.extract_mt_bench_rating_judgment",
        ],
    ),
    "templates.response_assessment.rating.mt_bench_multi_turn",
    overwrite=True,
)
