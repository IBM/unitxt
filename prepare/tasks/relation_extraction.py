from unitxt.blocks import FormTask
from unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs={"text": "str", "relation_type": "str"},
        outputs={
            "relation_entity_subject1": "List[str]",
            "relation_subject1_span_begins": "List[str]",
            "relation_subject1_span_ends": "List[str]",
            "relation_entity_subject2": "List[str]",
            "relation_subject2_span_begins": "List[str]",
            "relation_subject2_span_ends": "List[str]",
            "relation_type": "List[str]",
        },
        prediction_type="List[Tuple[str,str,str]]",
        metrics=["metrics.relation_extraction"],
        augmentable_inputs=["text"],
    ),
    "tasks.relation_extraction.single_relation_type",
    overwrite=True,
)

add_to_catalog(
    FormTask(
        inputs={"text": "str", "relations_types": "List[str]"},
        outputs={
            "relation_entity_subject1": "List[str]",
            "relation_subject1_span_begins": "List[str]",
            "relation_subject1_span_ends": "List[str]",
            "relation_entity_subject2": "List[str]",
            "relation_subject2_span_begins": "List[str]",
            "relation_subject2_span_ends": "List[str]",
            "relation_type": "List[str]",
        },
        prediction_type="List[Tuple[str,str,str]]",
        metrics=["metrics.relation_extraction"],
        augmentable_inputs=["text"],
    ),
    "tasks.relation_extraction.all_relations_types",
    overwrite=True,
)

add_to_catalog(
    FormTask(
        inputs={"input" :"str", "relation_types":"List[str]"},
        outputs={
            "date":"List[str]",
            "employee_number":"List[str]",
            "relation_employee_percent":"List[str]",
            "relation_job_role":"List[str]",
            "relation_company_department":"List[str]",
            "relation_geography":"List[str]",
        },
        prediction_type="List[Dict]",
        metrics=["metrics.relation_extraction"],
        augmentable_inputs=["text"],
    ),
    "tasks.relation_extraction.n-ary_relations_types",
    overwrite=True,
)