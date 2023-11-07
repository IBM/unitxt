from src.unitxt.templates import TemplatesDict, TemplatesList
from src.unitxt.blocks import InputOutputTemplate
from src.unitxt.catalog import add_to_catalog

templates = {
    "mmlu": """The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:""".strip(),
    "helm": """The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:""".strip(),
    "lm_eval_harness": """Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:""".strip(),
    "fm_eval": """The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:""".strip(),
}


# MMLU_TEMPLATES = TemplatesDict(
#     {key: InputOutputTemplate(input_format=val, output_format="{label}", postprocessors=["processors.first_character"]) for key, val in templates.items()}
# )

for k, v in templates.items():
    template = InputOutputTemplate(input_format=v, output_format="{label}", postprocessors=["processors.first_character"])
    add_to_catalog(template, f"templates.qa.multiple_choice.original.{k}", overwrite=True)

# with context


def replace_if_context_not_there(s, oldvalue, newvalue):
    if "{context}" in s:
        return s
    else:
        return s.replace(oldvalue, newvalue)


templates_with_context = {
    key: replace_if_context_not_there(
        replace_if_context_not_there(
            val,
            "Question:",
            "Context: {context}\nQuestion:",
        ),
        "{sentence1}",
        "{context}\n{sentence1}",
    )
    for key, val in templates.items()
}

# CONTEXT_MMLU_TEMPLATES = TemplatesDict(
#     {
#         key: InputOutputTemplate(input_format=val, output_format="{label}")
#         for key, val in templates_with_context.items()
#     }
# )

for k, v in templates_with_context.items():
    template = InputOutputTemplate(input_format=v, output_format="{label}", postprocessors=["processors.first_character"])
    add_to_catalog(template, f"templates.qa.multiple_choice.context.{k}", overwrite=True)


# context no intro
templates_context_no_intro = {
    key: val.replace("The following are multiple choice questions (with answers) about {topic}.", "").strip()
    for key, val in templates_with_context.items()
}

# CONTEXT_MMLU_TEMPLATES_NO_INTRO = TemplatesDict(
#     {
#         key: InputOutputTemplate(input_format=val, output_format="{label}")
#         for key, val in templates_context_no_intro.items()
#     }
# )

for k, v in templates_context_no_intro.items():
    template = InputOutputTemplate(input_format=v, output_format="{label}", postprocessors=["processors.first_character"])
    add_to_catalog(template, f"templates.qa.multiple_choice.context_no_intro.{k}", overwrite=True)

# no intro
templates_no_intro = {
    key: val.replace("The following are multiple choice questions (with answers) about {topic}.", "").strip()
    for key, val in templates.items()
}

# MMLU_TEMPLATES_NO_INTRO = TemplatesDict(
#     {key: InputOutputTemplate(input_format=val, output_format="{label}") for key, val in templates_no_intro.items()}
# )

for k, v in templates_no_intro.items():
    template = InputOutputTemplate(input_format=v, output_format="{label}", postprocessors=["processors.first_character"])
    add_to_catalog(template, f"templates.qa.multiple_choice.no_intro.{k}", overwrite=True)

# add template aggragations
template_list = []
for template_family in ["original", "context_no_intro", "no_intro", "context"]:
    family_list = [
        f"templates.qa.multiple_choice.{template_family}.{template_type}" for template_type in templates.keys()
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
