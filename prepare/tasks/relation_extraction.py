from unitxt.blocks import FormTask
from unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs={"text": "str", "relation_type": "str"},
        outputs={
            "relation_entity_subject1": "List[str]",
            "relation_type": "List[str]",
            "relation_entity_subject2": "List[str]",
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
            "relation_type": "List[str]",
            "relation_entity_subject2": "List[str]",
        },
        prediction_type="List[Tuple[str,str,str]]",
        metrics=["metrics.relation_extraction"],
        augmentable_inputs=["text"],
    ),
    "tasks.relation_extraction.all_relations_types",
    overwrite=True,
)
