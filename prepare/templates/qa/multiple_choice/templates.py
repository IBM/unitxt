from src.unitxt.catalog import add_to_catalog
from src.unitxt.templates import InputOutputChoicesTemplate, TemplatesList

output_format = "{answer}"

# MMLU (original)

input_format = "The following are multiple choice questions (with answers) about {topic}.\n{question}\nAnswers:\n{choices}\nAnswer:"
add_to_catalog(
    InputOutputChoicesTemplate(
        input_format=input_format,
        output_format=output_format,
        target_field="answer",
        choices_seperator="\n",
        enumerator="capitals",
        target_with_choice_text=False,
        target_with_choice_numeral=True,
    ),
    "templates.qa.multiple_choice.topical.mmlu",
    overwrite=True,
)

input_format = "The following are multiple choice questions (with answers) about {topic}.\n{context}\n{question}\nAnswers:\n{choices}\nAnswer:"
add_to_catalog(
    InputOutputChoicesTemplate(
        input_format=input_format,
        output_format=output_format,
        target_field="answer",
        choices_seperator="\n",
        enumerator="capitals",
        target_with_choice_text=False,
        target_with_choice_numeral=True,
    ),
    "templates.qa.multiple_choice.contextual_with_topic.mmlu",
    overwrite=True,
)

# HELM

input_format = "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {question}\nAnswers:\n{choices}\nAnswer:"
add_to_catalog(
    InputOutputChoicesTemplate(
        input_format=input_format,
        output_format=output_format,
        target_field="answer",
        choices_seperator="\n",
        enumerator="capitals",
        target_with_choice_text=False,
        target_with_choice_numeral=True,
    ),
    "templates.qa.multiple_choice.topical.helm",
    overwrite=True,
)

input_format = "The following are multiple choice questions (with answers) about {topic}.\n\nContext: {context}\nQuestion: {question}\nAnswers:\n{choices}\nAnswer:"
add_to_catalog(
    InputOutputChoicesTemplate(
        input_format=input_format,
        output_format=output_format,
        target_field="answer",
        choices_seperator="\n",
        enumerator="capitals",
        target_with_choice_text=False,
        target_with_choice_numeral=True,
    ),
    "templates.qa.multiple_choice.contextual_with_topic.helm",
    overwrite=True,
)

# lm_eval_harness

input_format = "Question: {question}\nChoices:\n{choices}\nAnswer:"
add_to_catalog(
    InputOutputChoicesTemplate(
        input_format=input_format,
        output_format=output_format,
        target_field="answer",
        choices_seperator="\n",
        enumerator="capitals",
        target_with_choice_text=False,
        target_with_choice_numeral=True,
    ),
    "templates.qa.multiple_choice.lm_eval_harness",
    overwrite=True,
)

input_format = "Context: {context}\nQuestion: {question}\nChoices:\n{choices}\nAnswer:"
add_to_catalog(
    InputOutputChoicesTemplate(
        input_format=input_format,
        output_format=output_format,
        target_field="answer",
        choices_seperator="\n",
        enumerator="capitals",
        target_with_choice_text=False,
        target_with_choice_numeral=True,
    ),
    "templates.qa.multiple_choice.contextual.lm_eval_harness",
    overwrite=True,
)

# fm_eval

input_format = "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {question}\nChoose from {numerals}\nAnswers:\n{choices}\nAnswer:"
add_to_catalog(
    InputOutputChoicesTemplate(
        input_format=input_format,
        output_format=output_format,
        target_field="answer",
        choices_seperator="\n",
        enumerator="capitals",
        add_numerals_as_field="numerals",
        target_with_choice_text=False,
        target_with_choice_numeral=True,
    ),
    "templates.qa.multiple_choice.topical.fm_eval",
    overwrite=True,
)

input_format = "The following are multiple choice questions (with answers) about {topic}.\n\nContext: {context}\nQuestion: {question}\nChoose from {numerals}\nAnswers:\n{choices}\nAnswer:"
add_to_catalog(
    InputOutputChoicesTemplate(
        input_format=input_format,
        output_format=output_format,
        target_field="answer",
        choices_seperator="\n",
        enumerator="capitals",
        add_numerals_as_field="numerals",
        target_with_choice_text=False,
        target_with_choice_numeral=True,
    ),
    "templates.qa.multiple_choice.contextual_with_topic.fm_eval",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.qa.multiple_choice.contextual.lm_eval_harness",
        ]
    ),
    "templates.qa.multiple_choice.contextual.all",
    overwrite=True,
)

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
            "templates.qa.multiple_choice.topical.fm_eval",
            "templates.qa.multiple_choice.topical.mmlu",
            "templates.qa.multiple_choice.topical.helm",
        ]
    ),
    "templates.qa.multiple_choice.topical.all",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.qa.multiple_choice.lm_eval_harness",
        ]
    ),
    "templates.qa.multiple_choice.all",
    overwrite=True,
)
