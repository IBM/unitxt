from unitxt.catalog import add_to_catalog
from unitxt.templates import InputOutputTemplate, TemplatesList

add_to_catalog(
    InputOutputTemplate(
        input_format=(
            'Sentence 1: "{text_a}"\n'
            'Sentence 2: "{text_b}"\n'
            "Is sentence 2 true, based on sentence 1?\n"
        ),
        output_format="ANS:\n{label}",
        postprocessors=[
            "processors.get_string_after_colon",
            "processors.take_first_non_empty_line",
            "processors.lower_case_till_punc",
        ],
    ),
    "templates.classification.multi_class.relation.truthfulness.flan_1",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format=('Problem: If "{text_a}", does it follow that "{text_b}"?\n'),
        output_format="Answer: {label}\n",
        postprocessors=[
            "processors.get_string_after_colon",
            "processors.take_first_non_empty_line",
            "processors.lower_case_till_punc",
        ],
    ),
    "templates.classification.multi_class.relation.truthfulness.flan_2",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format=('Input: Can we say "{text_b}" if "{text_a}"?\n'),
        output_format="{label}\n",
        target_prefix="Output:",
        postprocessors=[
            "processors.take_first_non_empty_line",
            "processors.lower_case_till_punc",
        ],
    ),
    "templates.classification.multi_class.relation.truthfulness.flan_3",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format=(
            'input question: Is it true that "{text_b}" if "{text_a}" is true?'
        ),
        output_format="{label}",
        target_prefix="output answer:",
        postprocessors=[
            "processors.take_first_non_empty_line",
            "processors.lower_case_till_punc",
        ],
    ),
    "templates.classification.multi_class.relation.truthfulness.flan_4",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format=(
            'Problem: Sentence: "{text_a}";\n' 'Another sentence: "{text_b}"?'
        ),
        output_format="{label}",
        target_prefix="A: ",
        postprocessors=[
            "processors.take_first_non_empty_line",
            "processors.lower_case_till_punc",
        ],
    ),
    "templates.classification.multi_class.relation.truthfulness.flan_5",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format=(
            'question: "{text_a}" is true.\n' 'So, is "{text_b}" true as well?\n'
        ),
        output_format="{label}\n",
        target_prefix="prediction: ",
        postprocessors=[
            "processors.take_first_non_empty_line",
            "processors.lower_case_till_punc",
        ],
    ),
    "templates.classification.multi_class.relation.truthfulness.flan_6",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format=(
            "Question:\n"
            'SA: "{text_a}"\n\n'
            'SB: "{text_b}"\n'
            "\n"
            "Is SB true, based on SA?\n"
        ),
        output_format="{label}",
        target_prefix="Answer: ",
        postprocessors=[
            "processors.take_first_non_empty_line",
            "processors.lower_case_till_punc",
        ],
    ),
    "templates.classification.multi_class.relation.truthfulness.flan_7",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            f"templates.classification.multi_class.relation.truthfulness.flan_{i}"
            for i in range(1, 8)
        ]
    ),
    "templates.classification.multi_class.relation.truthfulness.all",
    overwrite=True,
)
