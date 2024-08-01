from unitxt.blocks import FormTask
from unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs={"text": "List[str]", "relation_type": "str"},
        outputs={
            "subject_mentions": "List[str]",
            "label": "List[str]",
            "object_mentions": "List[str]",
            "text": 'List[str]'
        },
        prediction_type="List[Tuple[str,str,str]]",
        metrics=["metrics.specified_relation_extraction"],
        augmentable_inputs=["text"],
    ),
    "tasks.relation_extraction.single_relation_type_for_pair_no_spans",
    overwrite=True,
)

add_to_catalog(
    FormTask(
        inputs={"text": "List[str]", "relation_types": "List[str]"},
        outputs={
            "subject_mentions": "List[str]",
            "labels": "List[str]",
            "object_mentions": "List[str]",
            "text": "List[str]",
        },
        prediction_type="List[Tuple[str,str,str]]",
        metrics=["metrics.specified_relation_extraction"],
        augmentable_inputs=["text"],
    ),
    "tasks.relation_extraction.all_relation_types_for_pairs_no_spans",
    overwrite=True,
)

# add_to_catalog(
#     FormTask(
#         inputs={"text": "str", "relation_type": "str"},
#         outputs={
#             "subject_mentions": "List[str]",
#             "relation_type": "List[str]",
#             "object_mentions": "List[str]",
#             "subjects_starts": "List[int]",
#             "subjects_ends": "List[int]",
#             "objects_starts": "List[int]",
#             "objects_ends": "List[int]",
#         },
#         prediction_type="List[Tuple[str,str,str]]",
#         metrics=["metrics.relation_extraction"],
#         augmentable_inputs=["text"],
#     ),
#     "tasks.relation_extraction.single_relation_type_for_pair_w_spans",
#     overwrite=True,
# )

# add_to_catalog(
#     FormTask(
#         inputs={"text": "str", "relations_types": "List[str]"},
#         outputs={
#             "subject_mentions": "List[str]",
#             "relation_type": "List[str]",
#             "object_mentions": "List[str]",
#             "subjects_starts": "List[int]",
#             "subjects_ends": "List[int]",
#             "objects_starts": "List[int]",
#             "objects_ends": "List[int]",
#         },
#         prediction_type="List[Tuple[str,str,str]]",
#         metrics=["metrics.relation_extraction"],
#         augmentable_inputs=["text"],
#     ),
#     "tasks.relation_extraction.all_relations_types_for_pairs_w_spans",
#     overwrite=True,
# )

# add_to_catalog(
#     FormTask(
#         inputs={"input" :"str", "relation_types":"List[str]"},
#         outputs={
#             "date":"List[str]",
#             "employee_number":"List[str]",
#             "relation_employee_percent":"List[str]",
#             "relation_job_role":"List[str]",
#             "relation_company_department":"List[str]",
#             "relation_geography":"List[str]",
#         },
#         prediction_type="List[Dict]",
#         metrics=["metrics.relation_extraction"],
#         augmentable_inputs=["text"],
#     ),
#     "tasks.relation_extraction.n-ary_relations_types",
#     overwrite=True,
# )