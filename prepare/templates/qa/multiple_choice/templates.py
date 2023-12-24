from src.unitxt.catalog import add_to_catalog
from src.unitxt.templates import MultipleChoiceTemplate, TemplatesList

templates = {
    "mmlu": "The following are multiple choice questions (with answers) about {topic}.\n{question}.\nAnswers: \n{choices}.\nAnswer:",
    "helm": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {question}.\nAnswers: \n{choices}.\nAnswer:",
    "fm_eval": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {question}\nChoose from {numerals}\nAnswers: \n{choices}\nAnswer:".strip(),
    "lm_eval_harness": "{question}\n{choices}\nAnswer:" "",
}

for k, v in templates.items():
    template = MultipleChoiceTemplate(
        input_format=v,
        target_field="answer",
        choices_seperator="\n",
        postprocessors=["processors.first_character"],
    )
    add_to_catalog(
        template, f"templates.qa.multiple_choice.original.{k}", overwrite=True
    )

# https://github.com/EleutherAI/lm-evaluation-harness/blob/9e03d9d024be9bc3e92f8c63b5595c1e12c119da/lm_eval/tasks/mmlu/default/_default_template_yaml
# from lm-eval-harness
#'The following are multiple choice questions (with answers) about abstract algebra.\n\nFind the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\nA. 0\nB. 4\nC. 2\nD. 6\nAnswer:'
input_format = """
The following are multiple choice questions (with answers) about {topic}.

{question}
{choices}
Answer:
""".strip()

add_to_catalog(
    MultipleChoiceTemplate(
        input_format=input_format,
        target_field="answer",
        choices_seperator="\n",
        postprocessors=["processors.first_character"],
    ),
    "templates.qa.multiple_choice.original.lm_eval_harness",
    overwrite=True,
)


# input_format = "Question: {question}\nChoices:\n{choices}\nAnswer:"
# add_to_catalog(
#     MultipleChoiceTemplate(
#         input_format=input_format,
#         target_field="answer",
#         choices_seperator="\n",
#         postprocessors=["processors.first_character"],
#     ),
#     "templates.qa.multiple_choice.lm_eval_harness",
#     overwrite=True,
# )

# with context


def replace_if_context_not_there(s, oldvalue, newvalue):
    if "{context}" in s:
        return s

    return s.replace(oldvalue, newvalue)


templates_with_context = {
    key: replace_if_context_not_there(
        replace_if_context_not_there(
            val,
            "Question:",
            "Context: {context}\nQuestion:",
        ),
        "{question}",
        "{context}\n{question}",
    )
    for key, val in templates.items()
}

for k, v in templates_with_context.items():
    template = MultipleChoiceTemplate(
        input_format=v,
        target_field="answer",
        choices_seperator="\n",
        postprocessors=["processors.first_character"],
    )
    add_to_catalog(
        template, f"templates.qa.multiple_choice.context.{k}", overwrite=True
    )


# context no intro
templates_context_no_intro = {
    key: val.replace(
        "The following are multiple choice questions (with answers) about {topic}.", ""
    ).strip()
    for key, val in templates_with_context.items()
}

for k, v in templates_context_no_intro.items():
    template = MultipleChoiceTemplate(
        input_format=v,
        target_field="answer",
        choices_seperator="\n",
        postprocessors=["processors.first_character"],
    )
    add_to_catalog(
        template, f"templates.qa.multiple_choice.context_no_intro.{k}", overwrite=True
    )

# no intro
templates_no_intro = {
    key: val.replace(
        "The following are multiple choice questions (with answers) about {topic}.", ""
    ).strip()
    for key, val in templates.items()
}

for k, v in templates_no_intro.items():
    template = MultipleChoiceTemplate(
        input_format=v,
        target_field="answer",
        choices_seperator="\n",
        postprocessors=["processors.first_character"],
    )
    add_to_catalog(
        template, f"templates.qa.multiple_choice.no_intro.{k}", overwrite=True
    )

# add template aggragations
template_list = []
for template_family in ["original", "context_no_intro", "no_intro", "context"]:
    family_list = [
        f"templates.qa.multiple_choice.{template_family}.{template_type}"
        for template_type in templates.keys()
    ]
    add_to_catalog(
        TemplatesList(family_list),
        f"templates.qa.multiple_choice.{template_family}.all",
        overwrite=True,
    )
    template_list.extend(family_list)

add_to_catalog(
    TemplatesList(template_list),
    "templates.qa.multiple_choice.all",
    overwrite=True,
)


output_format = "{answer}"

# MMLU (original)

input_format = "The following are multiple choice questions (with answers) about {topic}.\n{question}\nAnswers:\n{choices}\nAnswer:"
add_to_catalog(
    MultipleChoiceTemplate(
        input_format=input_format,
        target_field="answer",
        choices_seperator="\n",
        postprocessors=["processors.first_character"],
    ),
    "templates.qa.multiple_choice.with_topic.mmlu",
    overwrite=True,
)

input_format = "The following are multiple choice questions (with answers) about {topic}.\n{context}\n{question}\nAnswers:\n{choices}\nAnswer:"
add_to_catalog(
    MultipleChoiceTemplate(
        input_format=input_format,
        target_field="answer",
        choices_seperator="\n",
        postprocessors=["processors.first_character"],
    ),
    "templates.qa.multiple_choice.contextual_with_topic.mmlu",
    overwrite=True,
)

# HELM

input_format = "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {question}\nAnswers:\n{choices}\nAnswer:"
add_to_catalog(
    MultipleChoiceTemplate(
        input_format=input_format,
        target_field="answer",
        choices_seperator="\n",
        postprocessors=["processors.first_character"],
    ),
    "templates.qa.multiple_choice.with_topic.helm",
    overwrite=True,
)

input_format = "The following are multiple choice questions (with answers) about {topic}.\n\nContext: {context}\nQuestion: {question}\nAnswers:\n{choices}\nAnswer:"
add_to_catalog(
    MultipleChoiceTemplate(
        input_format=input_format,
        target_field="answer",
        choices_seperator="\n",
        postprocessors=["processors.first_character"],
    ),
    "templates.qa.multiple_choice.contextual_with_topic.helm",
    overwrite=True,
)

# # lm_eval_harness

# input_format = "Question: {question}\nChoices:\n{choices}\nAnswer:"
# add_to_catalog(
#     MultipleChoiceTemplate(
#         input_format=input_format,
#         target_field="answer",
#         choices_seperator="\n",
#         postprocessors=["processors.first_character"],
#     ),
#     "templates.qa.multiple_choice.lm_eval_harness",
#     overwrite=True,
# )

# input_format = "Context: {context}\nQuestion: {question}\nChoices:\n{choices}\nAnswer:"
# add_to_catalog(
#     MultipleChoiceTemplate(
#         input_format=input_format,
#         target_field="answer",
#         choices_seperator="\n",
#         postprocessors=["processors.first_character"],
#     ),
#     "templates.qa.multiple_choice.contextual.lm_eval_harness",
#     overwrite=True,
# )

# fm_eval

input_format = "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {question}\nChoose from {numerals}\nAnswers:\n{choices}\nAnswer:"
add_to_catalog(
    MultipleChoiceTemplate(
        input_format=input_format,
        target_field="answer",
        choices_seperator="\n",
        add_numerals_as_field="numerals",
        postprocessors=["processors.first_character"],
    ),
    "templates.qa.multiple_choice.with_topic.fm_eval",
    overwrite=True,
)

input_format = "The following are multiple choice questions (with answers) about {topic}.\n\nContext: {context}\nQuestion: {question}\nChoose from {numerals}\nAnswers:\n{choices}\nAnswer:"
add_to_catalog(
    MultipleChoiceTemplate(
        input_format=input_format,
        target_field="answer",
        choices_seperator="\n",
        add_numerals_as_field="numerals",
        postprocessors=["processors.first_character"],
    ),
    "templates.qa.multiple_choice.contextual_with_topic.fm_eval",
    overwrite=True,
)

# add_to_catalog(
#     TemplatesList(
#         [
#             "templates.qa.multiple_choice.contextual.lm_eval_harness",
#         ]
#     ),
#     "templates.qa.multiple_choice.contextual.all",
#     overwrite=True,
# )

add_to_catalog(
    TemplatesList(
        [
            "templates.qa.multiple_choice.contextual_with_topic.fm_eval",
            "templates.qa.multiple_choice.contextual_with_topic.mmlu",
            "templates.qa.multiple_choice.contextual_with_topic.helm",
        ]
    ),
    "templates.qa.multiple_choice.contextual_with_topic.all",
    overwrite=True,
)


add_to_catalog(
    TemplatesList(
        [
            "templates.qa.multiple_choice.with_topic.fm_eval",
            "templates.qa.multiple_choice.with_topic.mmlu",
            "templates.qa.multiple_choice.with_topic.helm",
        ]
    ),
    "templates.qa.multiple_choice.with_topic.all",
    overwrite=True,
)

# add_to_catalog(
#     TemplatesList(
#         [
#             "templates.qa.multiple_choice.lm_eval_harness",
#         ]
#     ),
#     "templates.qa.multiple_choice.all",
#     overwrite=True,
# )
