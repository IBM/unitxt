from unitxt.templates import InputOutputTemplate, MultiLabelTemplate, TemplatesList, TupleTemplate
from unitxt.catalog import add_to_catalog


import fm_eval.runnables.local_catalogs
from fm_eval.runnables.local_catalogs import add_to_private_catalog
from prepare.templates.template_utils import add_templates_to_private_catalog


######## PAIR RELATIONS ALL

all_relations_for_pair_templates = {
    "en": {
        "extract_all_relations": "From the following text, extract relations of one of the following types: {relation_types}.\nText: {text}\n",
        "empty": "{text}",
    },
}

for language in all_relations_for_pair_templates.keys():
    all_entities_for_pair_templates_list = []
    suffix = f".{language}" if (language != "en") else ""
    all_entities_for_pair_templates_list.extend(
        add_templates_to_private_catalog(
            TupleTemplate,
            tuple_fields=["subject_mentions", "labels", "object_mentions"],
            template_name_to_input_format=all_relations_for_pair_templates[language],
            # output_format="{labels}",
            postprocessors=[
                "processors.take_first_non_empty_line",
                "processors.lower_case_till_punc",
                "processors.to_list_of_tuples_from_string_by_comma",
            ],
            task_name=f"relation_extraction.all_relation_types{suffix}",
        )
    )
    add_to_private_catalog(
        TemplatesList(all_entities_for_pair_templates_list),
        f"templates.relation_extraction.all_relation_types.all",
    )

######### PAIR RELATIONS SINGLE

# single_relation_types = ["employedBy", "managerOf", "basedIn"]

# single_relation_types_to_lang = {
#     "en": {"employedBy": "employedBy", "managerOf": "managerOf", "basedIn": "basedIn"},
# }

# none_to_lang = {
#     "en": "None",
# }

# single_relation_templates = {
#     "en": {
#         "extract_relations_of_type": "From the following sentence, extract two entities that form relation of the following type: {relation_type}.\nText: {text}\n",
#         "empty": "{text}",
#     },
# }

# for single_relation_type in single_relation_types:
#     for language in single_relation_templates.keys():
#         single_relation_templates_list = []
#         suffix = f".{language}" if (language != "en") else ""
#         single_relation_templates_list.extend(
#             add_templates_to_private_catalog(
#                 TupleTemplate,
#                 template_name_to_input_format=single_relation_templates[language],
#                 output_format="{label}",
#                 empty_label=none_to_lang[language],
#                 postprocessors=[
#                     "processors.take_first_non_empty_line",
#                     "processors.lower_case_till_punc",
#                     "processors.to_list_by_comma",
#                 ],
#                 task_name=f"relation_extraction.single_relation_type.{single_relation_type.lower()}{suffix}",
#             )
#         )
#         add_to_private_catalog(
#             TemplatesList(single_relation_templates_list),
#             f"templates.relation_extraction.single_relation_type.{single_relation_type.lower()}{suffix}.all",
#         )


# n_ary_relation_subtypes = [
#     "DATE",
#     "EMPLOYEE_NUMBER" "EMPLOYEE_PERCENT",
#     "JOB_ROLE",
#     "COMPANY_DEPARTMENT",
#     "GEOGRAPHY",
# ]

# n_ary_relation_templates = {
#     "en": {
#         "extract_n_ary_relations_of_entities": f"From the following sentence, collect all sets of {n_ary_relation_subtypes} that are related."
#     }
# }

# for language in n_ary_relation_templates.keys():
#     all_entities_for_pair_templates_list = []
#     suffix = f".{language}" if (language != "en") else ""
#     all_entities_for_pair_templates_list.extend(
#         add_templates_to_private_catalog(
#             SpanLabelingTemplate,
#             template_name_to_input_format=n_ary_relation_templates[language],
#             task_name=f"relation_extraction.extract_n_ary_relations{suffix}",
#         )
#     )
#     add_to_private_catalog(
#         TemplatesList(single_relation_templates_list),
#         f"templates.relation_extraction.extract_n_ary_relations.{single_relation_type.lower()}{suffix}.all",
#     )
